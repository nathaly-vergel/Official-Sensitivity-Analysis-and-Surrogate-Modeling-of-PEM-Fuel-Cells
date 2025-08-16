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
from src.FE.FE_formatting import load_parameter_ranges

# ----------------------------------------------------------------------
# Unique color palette for all input parameters
# ----------------------------------------------------------------------
# Load parameter names
param_config,parameter_ranges, parameter_names= load_parameter_ranges('../configs/param_config.yaml')

# Add 'ifc' if it's ever included in plots
if 'ifc' not in parameter_names:
    parameter_names.append('ifc')

# Define a consistent color map for features
COLORS = OmegaConf.load('../configs/colors_cfg.yaml')['COLORS']
FEATURE_COLOR_MAP = {feat: COLORS[i % len(COLORS)] for i, feat in enumerate(parameter_names)}

# ----------------------------------------------------------------------

def plot_shap_bar(shap_df, top_n=13):
    """
    Plot a horizontal bar chart of mean absolute SHAP values for the top features.

    Parameters
    ----------
    shap_df : pd.DataFrame
        DataFrame containing 'feature' and 'mean_abs_shap' columns.
    top_n : int
        Number of top features to display.
    """
    plt.figure(figsize=(8, 5))
    features = shap_df["feature"][:top_n][::-1]
    colors = [FEATURE_COLOR_MAP.get(f, "#cccccc") for f in features]
    plt.barh(features, shap_df["mean_abs_shap"][:top_n][::-1], color=colors)
    plt.xlabel("Mean(|SHAP value|)")
    plt.title("SHAP Feature Importance")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_sobol_index_convergence(sobol_results, index_type="ST", top_k=8, title=None, log_x=False, region=None, step=128, max_N=1024):
    """
    Plot how each feature's Sobol index (S1 or ST) changes as sample size N increases.

    Parameters
    ----------
    sobol_results : dict
        Output from run_sobol_convergence_analysis(). Maps N → dict with key 'sobol_df'.
    index_type : str
        Which index to plot: "S1" or "ST".
    top_k : int
        Number of top features to show based on the highest N.
    title : str or None
        Custom plot title.
    log_x : bool
        Use log scale on the x-axis.
    region : str or None
        Region name to include in the title.
    step : int
        Step size used during convergence (used for ticks).
    max_N : int
        Max base N used during convergence (used for ticks).
    """
    assert index_type in ["S1", "ST"], "index_type must be 'S1' or 'ST'"

    max_N_avail = max(sobol_results)
    df_max = sobol_results[max_N_avail]["sobol_df"]
    top_features = df_max.sort_values(index_type, ascending=False)["feature"].head(top_k).tolist()

    N_vals = sorted(sobol_results.keys())
    data = {f: [] for f in top_features}

    for N in N_vals:
        df = sobol_results[N]["sobol_df"].set_index("feature")
        for f in top_features:
            value = df.loc[f, index_type] if f in df.index else np.nan
            data[f].append(value)

    plt.figure(figsize=(10, 6))
    for f in top_features:
        plt.plot(N_vals, data[f], marker="o", label=f, color=FEATURE_COLOR_MAP.get(f, "#cccccc"))

    plt.ylabel(f"{index_type} Sobol Index", fontsize = 16)
    plt.xlabel("Sample Size N", fontsize = 16)

    # Force x-tick alignment
    full_N_ticks = list(range(step, max_N + 1, step))
    plt.xticks(full_N_ticks, fontsize = 12)

    plt.ylim(0, 1.05)
    if title is None:
        title = f"{index_type} Index Convergence for {region.capitalize()} Region" if region else f"{index_type} Index Convergence"
    plt.title(title, fontsize=18)

    if log_x:
        plt.xscale("log")

    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend(loc="upper right", bbox_to_anchor=(1.18, 1.0), fontsize=11)
    plt.tight_layout()
    plt.show()

def plot_sobol_ranking(sobol_results, top_n=10):
    """
    Plot top first-order Sobol indices as horizontal bar plots for each N.

    Parameters
    ----------
    sobol_results : dict
        Dictionary where each key is N (sample size) and value contains 'sobol_df'.
    top_n : int
        Number of top features to display.
    """

    for N, result in sobol_results.items():
        plt.figure(figsize=(8, 4))
        df = result['sobol_df']
        top = df.sort_values("S1", ascending=False).head(top_n)
        colors = [FEATURE_COLOR_MAP.get(f, "#cccccc") for f in top["feature"][::-1]]
        plt.barh(top["feature"][::-1], top["S1"][::-1], color=colors)
        plt.title(f"Top {top_n} First-Order Sobol Indices (N={N})")
        plt.xlabel("S1")
        plt.grid(True)
        plt.tight_layout()
        plt.show()


def plot_top_k_rankings_across_regions(rank_sources, source_type="shap", top_k=13, figsize=(10, 6)):
    """
    Plot parameter ranking changes across regions.

    Parameters
    ----------
    rank_sources : dict
        Maps region name → either SHAP df or Si dict.
        SHAP df must have 'feature' and 'mean_abs_shap'.
        Si dict must have 'S1' or 'ST' and 'names'.
    source_type : str
        Either "shap", "S1", or "ST"
    top_k : int
        How many ranks to show (y-axis = 1 to top_k)
    figsize : tuple
        Size of the figure
    """
    regions = list(rank_sources.keys())
    all_features = set()
    ranks_per_region = {}

    for region, data in rank_sources.items():
        if source_type == "shap":
            df = data.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
            features = df["feature"].tolist()
        elif source_type in ["S1", "ST"]:
            scores = data[source_type]
            features = data["feature"]
            df = pd.DataFrame({"feature": features, "score": scores})
            df = df.sort_values("score", ascending=False).reset_index(drop=True)
            features = df["feature"].tolist()
        else:
            raise ValueError("Invalid source_type. Use 'shap', 'S1', or 'ST'.")

        all_features.update(features)
        ranks = {feat: i + 1 for i, feat in enumerate(features)}
        ranks_per_region[region] = ranks

    all_features = sorted(all_features)
    region_labels = list(rank_sources.keys())

    ranking_matrix = pd.DataFrame(index=all_features, columns=region_labels)
    for region in region_labels:
        for feat in all_features:
            ranking_matrix.loc[feat, region] = ranks_per_region[region].get(feat, np.nan)

    # Filter to top_k only
    filtered = ranking_matrix.apply(lambda row: any(r <= top_k for r in row if pd.notna(r)), axis=1)
    ranking_matrix = ranking_matrix[filtered]

    # --- Sort features by ranking in last region (e.g., mass transport) ---
    last_region = region_labels[-1]
    feature_order = ranking_matrix[last_region].sort_values().index.tolist()

    plt.figure(figsize=figsize)
    for feature in feature_order:
        y_vals = ranking_matrix.loc[feature].values.astype(float)
        color = FEATURE_COLOR_MAP.get(feature, "#cccccc")
        plt.plot(region_labels, y_vals, marker='o', label=feature, color=color)

    plt.gca().invert_yaxis()
    plt.xticks(rotation=0, fontsize = 13)
    plt.yticks(range(1, top_k + 1), fontsize=13)
    plt.xlabel("Current density region", fontsize=13)
    plt.ylabel("Ranking (1 = most important)", fontsize=13)
    plt.title(f"Top {top_k} Parameter Rankings across Regions ({source_type})", fontsize=16)

    # Sort legend handles to match line order
    handles, labels = plt.gca().get_legend_handles_labels()
    sorted_handles = [handles[labels.index(feat)] for feat in feature_order if feat in labels]
    plt.legend(sorted_handles, feature_order, bbox_to_anchor=(1.05, 1), loc='upper left',
               title="Parameter", fontsize=13)

    plt.tight_layout()
    plt.grid(True)
    plt.show()


def plot_sobol_region_barplot(
    df: pd.DataFrame,
    region: str,
    index_type: str = "S1",
    top_k: int = 10,
    figsize=(6, 4),
):
    """
    Plot top Sobol indices for a region as horizontal bars with confidence intervals.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with Sobol indices and columns: index_type and index_type+'_conf'.
    region : str
        Region name for title.
    index_type : str
        "S1" or "ST".
    top_k : int
        Number of top features to plot.
    figsize : tuple
        Figure size.
    """
    assert index_type in ["S1", "ST"], "Only 'S1' or 'ST' supported."

    # If feature column is missing, get it from index
    df = df.copy()
    if "feature" not in df.columns:
        df["feature"] = df.index

    df_sorted = df.sort_values(index_type, ascending=False).reset_index(drop=True)
    df_top = df_sorted.head(top_k).copy()
    df_top = df_top[::-1]  # Reverse for horizontal order

    features = df_top["feature"]
    values = df_top[index_type]
    confs = df_top[f"{index_type}_conf"]
    colors = [FEATURE_COLOR_MAP.get(f, "#cccccc") for f in features]

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.barh(features, values, color=colors, edgecolor="black")

    ax.errorbar(
        values,
        np.arange(len(features)),
        xerr=confs,
        fmt="none",
        ecolor="black",
        capsize=4,
        elinewidth=1
    )

    ax.set_xlabel(f"{index_type} Value", fontsize=12)
    ax.set_title(f"{index_type}-based Ranking: {region.capitalize()} region", fontsize=14)
    ax.grid(True, axis='x', linestyle="--", alpha=0.6)
    ax.tick_params(labelsize=11)
    plt.tight_layout()
    plt.show()