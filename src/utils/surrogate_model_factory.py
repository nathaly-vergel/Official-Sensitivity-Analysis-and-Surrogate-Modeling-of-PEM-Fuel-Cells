import os
import sys
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor
# Adjust the path to point to external/AlphaPEM
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
from src.validity.surrogate_model_validation import ensure_numeric_regions

def get_model_and_grid(model_name, random_state, grid_search):
    """
    Create a model instance and return it along with its hyperparameter grid.

    Parameters
    ----------
    model_name : str
        Identifier for the model type ('rf' for RandomForest, 'xgboost' for XGBRegressor).
    random_state : int
        Seed for reproducibility.
    grid_search : dict
        Configuration dictionary containing hyperparameter grids and (for XGBoost) fixed params.

    Returns
    -------
    model : estimator object
        Instantiated scikit-learn or XGBoost model.
    param_grid : dict
        Hyperparameter grid for tuning this model.
    """
    
    if model_name == 'rf':
        # Create a RandomForestRegressor with a fixed random seed
        model = RandomForestRegressor(random_state=random_state)

    elif model_name == 'xgboost':
        # Retrieve fixed parameters for XGBoost from the config
        params_xg = grid_search['xgboost']['params']
        # Create an XGBRegressor with fixed random seed and verbosity from config
        model = XGBRegressor(random_state=random_state, verbosity=params_xg['verbosity'])

    else:
        # Raise error if the model name is not recognized
        raise ValueError(f"[ERROR] Unknown model: {model_name}")
    
    # Retrieve the hyperparameter grid for this model from the config
    param_grid = dict(grid_search[model_name]['grid'])

    return model, param_grid


def compute_metrics(y_true, y_pred):
    """
    Compute regression performance metrics.

    Parameters
    ----------
    y_true : array-like
        Ground truth target values.
    y_pred : array-like
        Model predictions.

    Returns
    -------
    dict
        Dictionary containing:
        - r2  : Coefficient of determination
        - mse : Mean Squared Error
        - mae : Mean Absolute Error
    """
    return {
        'r2': r2_score(y_true, y_pred),
        'mse': mean_squared_error(y_true, y_pred),
        'mae': mean_absolute_error(y_true, y_pred)
    }


def pull(cfg, *keys, default=None):
    """
    Retrieve a nested value from a dictionary-like config using a sequence of keys.

    Parameters
    ----------
    cfg : dict
        Configuration dictionary.
    *keys : str
        Keys to traverse into the dictionary hierarchy.
    default : any, optional
        Value to return if any key is missing.

    Returns
    -------
    any
        The value found at the nested key path, or the default value if not found.

    Example
    -------
    cfg = {"cv": {"inner_splits": 5}}
    pull(cfg, "cv", "inner_splits")        # returns 5
    pull(cfg, "cv", "outer_splits", 3)     # returns 3 (default)
    """
    cur = cfg
    for k in keys:
        # If current level is None or key is missing, return default
        if cur is None or k not in cur:
            return default
        # Dive deeper into the nested dictionary
        cur = cur[k]
    return cur


def build_region_masks(x, regions_cfg):
    regions_cfg = ensure_numeric_regions(regions_cfg)
    x = np.asarray(x).ravel()  # works whether you pass X[:, -1] or a 1D array

    lo_a, hi_a = regions_cfg["activation"]
    lo_o, hi_o = regions_cfg["ohmic"]
    lo_m, hi_m = regions_cfg["mass_transport"]

    # left-closed / right-open to avoid overlap at boundaries
    masks = {
        "activation": x < hi_a,
        "ohmic": (x >= lo_o) & (x < hi_o),
        # choose open-ended or bounded; here we use bounded since your YAML has an upper limit
        "mass_transport": (x >= lo_m) & (x < hi_m),
    }
    return masks
