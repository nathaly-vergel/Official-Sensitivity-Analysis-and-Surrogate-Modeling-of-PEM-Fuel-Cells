import os
import sys
import json
import inspect
import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from datetime import datetime


sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
from src.utils.surrogate_model_factory import pull

def save_cv_results(
    X, y, groups,
    model_name, grid_search_cfg, core_training_cfg,cv_fn,regions_cfg=None,save_dir=None,run_name='model_run',
    inner_splits=None,outer_splits=None,random_state=None
):
    """
    Run a CV function (single or nested), train the best model on all data,
    and save metrics, hyperparameters, and the final model.

    Parameters
    ----------
    X : DataFrame or ndarray
    y : Series or ndarray
    groups : Series or ndarray
    model_name : str
        One of ["rf", "xgboost"]
    cv_fn : function
        A cross-validation function that returns either:
        - (model, best_params, metrics) for single CV
        - (metrics, best_params_list) for nested CV
    """

    


    random_state = random_state if random_state is not None else pull(core_training_cfg, "seed", default=42)
    outer_splits = outer_splits if outer_splits is not None else pull(core_training_cfg, "cv", "outer_splits", default=5)
    inner_splits = inner_splits if inner_splits is not None else pull(core_training_cfg, "cv", "inner_splits", default=3)
    timestamp_format = pull(core_training_cfg, "save", "timestamp_format", default="%Y%m%d_%H%M%S")
    save_dir = pull(core_training_cfg, "save", "save_dir", default="../results/surrogate_models/")

    save_dir = os.path.join(save_dir, model_name)
    os.makedirs(save_dir, exist_ok=True)
    print(f"[INFO] Running CV function: {cv_fn.__name__}")

    cv_fn_args = inspect.signature(cv_fn).parameters
    X_arr, y_arr, groups_arr = np.array(X), np.array(y), np.array(groups)

    # Decide if nested or single CV based on function signature
    if 'outer_splits' in cv_fn_args:
        # Nested CV
        metrics, best_params_list = cv_fn(
            X_arr, y_arr, groups_arr, X.columns,
            grid_search_cfg, core_training_cfg,regions_cfg,
            model_name=model_name,
            outer_splits=outer_splits,
            inner_splits=inner_splits,
            random_state=random_state
        )
        best_params = best_params_list[0]
    else:
        # Single CV
        model, best_params, metrics = cv_fn(
            X_arr, y_arr, groups_arr,
            grid_search_cfg, core_training_cfg,
            model_name=model_name,
            inner_splits=inner_splits,
            random_state=random_state
        )

    # Save metrics
    timestamp = datetime.now().strftime(timestamp_format)
    filename = f"{run_name}_{model_name}_{timestamp}_metrics.json"
    metrics_path = os.path.join(save_dir, filename)
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"[INFO] Saved metrics to {metrics_path}")

    # Save best hyperparameters
    filename = f"{run_name}_{model_name}_{timestamp}_best_params.json"
    params_path = os.path.join(save_dir, filename)
    with open(params_path, 'w') as f:
        json.dump(best_params, f, indent=4)
    print(f"[INFO] Saved best hyperparameters to {params_path}")

    # Retrain final model on all data using best parameters
    model_class = {
        'rf': RandomForestRegressor,
        'xgboost': XGBRegressor
    }.get(model_name)

    if model_class is None:
        raise ValueError(f"[ERROR] Unsupported model: {model_name}")

    final_model = model_class(random_state=random_state, **best_params)
    final_model.fit(X_arr, y_arr)

    # Save trained model
    filename = f"{run_name}_{model_name}_{timestamp}_final_model.pkl"
    model_path = os.path.join(save_dir, filename)
    joblib.dump(final_model, model_path)
    print(f"[INFO] Saved trained model to {model_path}")

    return final_model, best_params, metrics