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
from typing import Any, Mapping
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor
from itertools import product


def ensure_numeric_dataframe(X):
    """
    Ensure that all columns in a DataFrame are numeric (float or int).
    Converts object columns to float if possible.
    Raises an error if conversion fails.

    Parameters:
    - X: pd.DataFrame

    Returns:
    - pd.DataFrame with only numeric columns
    """
    import pandas as pd

    X_checked = X.copy()
    non_numeric_cols = X_checked.select_dtypes(exclude=['number']).columns

    if len(non_numeric_cols) > 0:
        print(f"[WARN] Found non-numeric columns: {list(non_numeric_cols)}")
        for col in non_numeric_cols:
            try:
                X_checked[col] = pd.to_numeric(X_checked[col])
                print(f"[INFO] Converted column '{col}' to numeric.")
            except Exception as e:
                raise ValueError(f"[ERROR] Failed to convert column '{col}' to numeric. Reason: {e}")
    else:
        print("[INFO] All columns are already numeric.")

    return X_checked


def to_plain(obj: Any):
    # Convert OmegaConf DictConfig/ListConfig to plain dict/list
    try:
        from omegaconf import OmegaConf
        # to_container handles nested structures
        return OmegaConf.to_container(obj, resolve=True)
    except Exception:
        return obj

def ensure_numeric_regions(regions_cfg):
    regions_cfg = to_plain(regions_cfg)  # <— key step

    if not isinstance(regions_cfg, Mapping):
        raise TypeError(f"regions_cfg must be a mapping, got {type(regions_cfg).__name__}")

    clean_cfg = {}
    for region, bounds in regions_cfg.items():
        if not isinstance(bounds, (list, tuple)):
            # after to_plain, these will be list/tuple
            raise TypeError(f"Bounds for '{region}' must be list/tuple, got {type(bounds).__name__}")
        if len(bounds) != 2:
            raise ValueError(f"Bounds for '{region}' must have exactly 2 values, got {bounds}")
        try:
            lo, hi = float(bounds[0]), float(bounds[1])
        except (TypeError, ValueError):
            raise ValueError(f"Bounds for '{region}' contain non-numeric values: {bounds}")
        clean_cfg[region] = [lo, hi]

    return clean_cfg