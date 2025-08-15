import gc
gc.collect()
import numpy as np
import pandas as pd
import os
import sys
from omegaconf import OmegaConf
import matplotlib.pyplot as plt
import json
import joblib
from SALib.sample.sobol import sample as sobol_sample
from SALib.analyze import sobol
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
from src.analysis.sensitivity import SensitivityAnalyzer
from src.utils.FE_formatting import expand_column_to_columns,parse_dependent_parameters,load_parameter_ranges

# ----------------------------------------------------------------------
# Unique color palette for all input parameters
# ----------------------------------------------------------------------
# Load parameter names
param_config,parameter_ranges, parameter_names= load_parameter_ranges('../configs/param_config.yaml')
parameter_names = list(param_config.keys())  

# Add 'ifc' if it's ever included in plots
if 'ifc' not in parameter_names:
    parameter_names.append('ifc')

# Define a consistent color map for features
COLORS = OmegaConf.load('../configs/colors_cfg.yaml')['COLORS']
FEATURE_COLOR_MAP = {feat: COLORS[i % len(COLORS)] for i, feat in enumerate(parameter_names)}

# ----------------------------------------------------------------------

def load_cv_results(save_dir='results', run_name='model_run'):
    """
    Load a previously saved model, best hyperparameters, and metrics.
    
    Parameters:
    - save_dir: folder containing the saved files
    - run_name: base name used when saving

    Returns:
    - model: trained sklearn or XGBoost model
    - best_params: dictionary of best hyperparameters
    - metrics: dictionary of evaluation metrics
    """

    model_path = os.path.join(save_dir, f"{run_name}_final_model.pkl")
    params_path = os.path.join(save_dir, f"{run_name}_best_params.json")
    metrics_path = os.path.join(save_dir, f"{run_name}_metrics.json")

    if not all(os.path.exists(p) for p in [model_path, params_path, metrics_path]):
        raise FileNotFoundError("[ERROR] One or more result files not found. Check save_dir and run_name.")

    model = joblib.load(model_path)

    with open(params_path, 'r') as f:
        best_params = json.load(f)

    with open(metrics_path, 'r') as f:
        metrics = json.load(f)

    print(f"[INFO] Loaded model from {model_path}")
    print(f"[INFO] Loaded hyperparameters from {params_path}")
    print(f"[INFO] Loaded metrics from {metrics_path}")

    return model, best_params, metrics


def save_FE_results(region_name,raw_shap,shap_df,sobol_results,save_dir="../results/xgboost",tag=None):

    """
    Save SHAP and Sobol feature importance results to disk for a given region.

    Parameters
    ----------
    region_name : str
        Name of the region to use in the output filenames.
    raw_shap: shap._explanation.Explanation
        SHAP Explanation object to be saved for further inspection or plotting.
    shap_df : pd.DataFrame
        DataFrame containing SHAP values and feature rankings.
    sobol_results : dict or None
        Dictionary containing Sobol analysis results (can be None if not used).
    save_dir : str
        Directory to save the output files to. Will be created if it doesn't exist.
    tag : str or None
        Optional tag to distinguish different versions (appended to filename).
    
    Saves
    -----
    - SHAP CSV: <save_dir>/xgb_<region_name>[_<tag>]_shap.csv
    - Sobol pickle: <save_dir>/xgb_<region_name>[_<tag>]_sobol_results.pkl
    """
    
    os.makedirs(save_dir, exist_ok=True)
    suffix = f"_{tag}" if tag else ""
    base = os.path.join(save_dir, f"xgb_{region_name}{suffix}")

    # Save SHAP
    shap_df.to_csv(f"{base}_shap.csv", index=False)
    print(f"[INFO] Saved SHAP ranking to {base}_shap.csv")

    # Save raw SHAP Explanation object
    raw_shap_path = f"{base}_raw_shap.pkl"
    joblib.dump(raw_shap, raw_shap_path)
    print(f"[INFO] Saved raw SHAP Explanation to {raw_shap_path}")

    # Save all Sobol results (DFs + Si + diagnostics) as one .pkl
    if sobol_results:
        sobol_path = f"{base}_sobol_results.pkl"
        joblib.dump(sobol_results, sobol_path)
        print(f"[INFO] Saved all Sobol results to {sobol_path}")


def load_FE_results(
    region_name,
    save_dir="../results/xgboost",
    tag=None
):
    """
    Load SHAP and Sobol results for a given region, including the raw SHAP Explanation object.

    Parameters
    ----------
    region_name : str
        Name of the region to load results for.
    save_dir : str
        Directory where results were saved.
    tag : str or None
        Optional tag to distinguish versions.

    Returns
    -------
    shap_df : pd.DataFrame
        DataFrame of SHAP values and rankings.
    raw_shap : shap.Explanation or None
        SHAP Explanation object (or None if not found).
    sobol_results : dict or None
        Dictionary of Sobol results (or None if not found).
    """
    suffix = f"_{tag}" if tag else ""
    base = os.path.join(save_dir, f"xgb_{region_name}{suffix}")

    shap_path = f"{base}_shap.csv"
    raw_shap_path = f"{base}_raw_shap.pkl"
    sobol_path = f"{base}_sobol_results.pkl"

    if not os.path.exists(shap_path):
        raise FileNotFoundError(f"[ERROR] SHAP file not found: {shap_path}")
    shap_df = pd.read_csv(shap_path)
    print(f"[INFO] Loaded SHAP ranking from {shap_path}")

    raw_shap = None
    if os.path.exists(raw_shap_path):
        raw_shap = joblib.load(raw_shap_path)
        print(f"[INFO] Loaded raw SHAP Explanation from {raw_shap_path}")
    else:
        print(f"[WARN] Raw SHAP Explanation not found: {raw_shap_path}")

    sobol_results = None
    if os.path.exists(sobol_path):
        sobol_results = joblib.load(sobol_path)
        print(f"[INFO] Loaded Sobol results from {sobol_path}")
    else:
        print(f"[WARN] Sobol results not found: {sobol_path}")

    return shap_df, raw_shap, sobol_results