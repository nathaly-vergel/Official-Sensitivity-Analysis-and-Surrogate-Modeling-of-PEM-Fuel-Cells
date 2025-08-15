import os
import sys
from omegaconf import OmegaConf
import matplotlib.pyplot as plt
import ast
import json
import joblib
import inspect
from datetime import datetime
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor
from itertools import product


# Adjust the path to point to external/AlphaPEM
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
from src.validity.validity_criteria import validate_polarization_curves
from src.utils.surrogate_model_factory import get_model_and_grid, compute_metrics,pull,build_region_masks

def nested_cv_train_with_groups(X, y, groups, feature_names,grid_search_cfg ,core_training_cfg,regions_cfg, model_name='rf', outer_splits=None, inner_splits=None,
    random_state=None, save_residuals=None,  ):
    
    
    random_state = random_state if random_state is not None else pull(core_training_cfg, "seed", default=42)
    outer_splits = outer_splits if outer_splits is not None else pull(core_training_cfg, "cv", "outer_splits", default=5)
    inner_splits = inner_splits if inner_splits is not None else pull(core_training_cfg, "cv", "inner_splits", default=3)
    scoring = pull(core_training_cfg, "cv", "scoring", default="r2")
    n_jobs = pull(core_training_cfg, "cv", "n_jobs", default=-1)
    gs_verbose = pull(core_training_cfg, "cv", "verbose", default=0)
    save_residuals = save_residuals if save_residuals is not None else pull(core_training_cfg, "save", "residuals", default=True)
    timestamp_format = pull(core_training_cfg, "save", "timestamp_format", default="%Y%m%d_%H%M%S")
    save_dir = pull(core_training_cfg, "save", "save_dir", default="../results/surrogate_models/")
    residuals_dir = os.path.join(save_dir, "residuals_by_fold")
    

    print("⚙️  Running nested CV with groups...")
    print(f"[INFO] Model: {model_name}")
    print(f"[INFO] Samples: {len(X)}, Groups: {len(np.unique(groups))}")
    print(f"[INFO] Outer Folds: {outer_splits}, Inner Folds: {inner_splits}")

    outer_cv = GroupKFold(n_splits=outer_splits)
    inner_cv = GroupKFold(n_splits=inner_splits)

    estimator, param_grid = get_model_and_grid(model_name, random_state, grid_search = grid_search_cfg)
    total_combinations = len(list(product(*param_grid.values())))
    print(f"[INFO] Hyperparameter grid: {total_combinations} combinations\n")

    metrics = {key: [] for key in [
        'r2', 'mse', 'mae',
        'r2_activation', 'mse_activation', 'mae_activation',
        'r2_ohmic', 'mse_ohmic', 'mae_ohmic',
        'r2_mass_transport', 'mse_mass_transport', 'mae_mass_transport'
    ]}
    best_params_list = []

    for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X, y, groups=groups), 1):
        print(f"\n🔁 Fold {fold}/{outer_splits}")
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        train_groups = groups[train_idx]

        grid_search = GridSearchCV(
            estimator, param_grid,
            cv=inner_cv.split(X_train, y_train, groups=train_groups),
            scoring=scoring, n_jobs=n_jobs, verbose=gs_verbose
        )
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        best_params_list.append(grid_search.best_params_)

        y_pred = best_model.predict(X_test)
        test_metrics = compute_metrics(y_test, y_pred)

        for key in ['r2', 'mse', 'mae']:
            metrics[key].append(test_metrics[key])

        # Region-based evaluations
        masks =  build_region_masks(X_test[:, -1] , regions_cfg)

        for region, mask in masks.items():
            if np.any(mask):
                region_metrics = compute_metrics(y_test[mask], best_model.predict(X_test[mask]))
            else:
                region_metrics = {'r2': np.nan, 'mse': np.nan, 'mae': np.nan}

            for key, val in region_metrics.items():
                metrics[f"{key}_{region}"].append(val)

        print(f"[INFO] Fold {fold} — R²: {test_metrics['r2']:.4f}, RMSE: {np.sqrt(test_metrics['mse']):.4f}, MAE: {test_metrics['mae']:.4f}")
    
        if save_residuals:
            os.makedirs(residuals_dir, exist_ok=True)
            
            # Convert X_test to DataFrame
            X_test_df = pd.DataFrame(X_test, columns=feature_names)

            # Create residuals DataFrame
            residuals_df = pd.DataFrame({
                'y_true': y_test,
                'y_pred': y_pred,
                'residual': y_test - y_pred
            })

            # Concatenate features
            residuals_df = pd.concat([residuals_df, X_test_df.reset_index(drop=True)], axis=1)

            # Save to CSV
            timestamp = datetime.now().strftime(timestamp_format)
            filename = f"residuals_fold_{fold}_{model_name}_{timestamp}.csv"
            residuals_df.to_csv(os.path.join(residuals_dir, filename), index=False)
            print(f"[INFO] Saved residuals and features to {os.path.join(residuals_dir, filename)}")

    return metrics, best_params_list



def single_cv_train_with_groups(X, y, groups,core_training_cfg, model_name='rf', inner_splits=None, random_state=None):
    """
    Perform a single grid search CV using GroupKFold to respect group structure.

    Returns
    -------
    best_model : trained model with best hyperparameters
    best_params : dict of best hyperparameters
    metrics : dict of evaluation metrics (R2, RMSE, MAE)
    """
    print(f"\n[INFO] Starting grid search CV for model: {model_name}")
    print(f"[INFO] Samples: {len(X)} | Unique groups: {len(np.unique(groups))} | Folds: {inner_splits}")

    random_state = random_state if random_state is not None else pull(core_training_cfg, "seed", default=42)
    inner_splits = inner_splits if inner_splits is not None else pull(core_training_cfg, "cv", "inner_splits", default=3)
    scoring = pull(core_training_cfg, "cv", "scoring", default="r2")
    n_jobs = pull(core_training_cfg, "cv", "n_jobs", default=-1)
    gs_verbose = pull(core_training_cfg, "cv", "verbose", default=0)
    
    inner_cv = GroupKFold(n_splits=inner_splits)
    model, param_grid = get_model_and_grid(model_name, random_state)

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring=scoring,
        cv=inner_cv.split(X, y, groups=groups),
        n_jobs=n_jobs,
        verbose=gs_verbose
    )
    grid_search.fit(X, y)

    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    print(f"[INFO] Best hyperparameters: {best_params}")

    y_pred = best_model.predict(X)
    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    mae = mean_absolute_error(y, y_pred)

    metrics = {
        'R2': r2,
        'RMSE': rmse,
        'MAE': mae
    }

    print(f"[INFO] Final performance on full data — R²: {r2:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}")

    return best_model, best_params, metrics


