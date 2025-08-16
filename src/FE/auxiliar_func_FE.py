import gc
gc.collect()
import numpy as np
import pandas as pd
import os
import sys
from omegaconf import OmegaConf
from SALib.sample.sobol import sample as sobol_sample
from SALib.analyze import sobol
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="SALib")


sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
from src.analysis.sensitivity import SensitivityAnalyzer
from src.FE.FE_formatting import parse_dependent_parameters,load_parameter_ranges



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

def run_sobol_convergence_analysis_for_region(SA,df, step=128, max_N=1024, index_type="S1"):
    """
    Perform Sobol sensitivity analysis for increasing sample sizes to check convergence 
    of indices by region.

    Parameters
    ----------
    SA : object
        Sensitivity analysis object with `aggregate_output_function` and `problem` dict.
    df : pandas.DataFrame
        Model output data.
    step : int, optional
        Minimum sample size (default 128).
    max_N : int, optional
        Maximum sample size (default 1024).
    index_type : str, optional
        "S1" for first-order or "ST" for total-order indices (default "S1").

    Returns
    -------
    list of dict
        Each element corresponds to a region; keys are sample sizes `N` and values 
        are dicts with `"sobol_df"` DataFrame of indices.
    """
    assert index_type in ["S1", "ST"], "index_type must be 'S1' or 'ST'"

    outputs = SA.aggregate_output_function(data=df, aggregation_method="AUC", by_regions=True)
    outputs_array = np.stack(outputs.to_numpy())

    convergence_regions =[]

    for i in range(outputs_array.shape[1]):
        Y = outputs_array[:,i]

        D = SA.problem["num_vars"]
        samples_per_N = 2 * D + 2
        N_values = [2**i for i in range(int(np.log2(step)), int(np.log2(max_N)) + 1)]

        sobol_convergence = {}

        for N_i in N_values:
            end_idx = N_i * samples_per_N
            if end_idx > len(Y):
                print(f"[SKIP] Not enough samples for N={N_i}. Needed {end_idx}, but got {len(Y)}.")
                continue

            Y_subset = Y[:end_idx]

            try:
                Si = sobol.analyze(SA.problem, Y_subset, calc_second_order=True, print_to_console=False)
                df_all = Si.to_df()
                df_si = df_all[0] if index_type == "ST" else df_all[1]
                df_si = df_si.copy()
                df_si["feature"] = SA.problem["names"]
                sobol_convergence[N_i] = {"sobol_df": df_si}
            except Exception as e:
                print(f"[FAIL] Sobol failed at N={N_i}: {e}")
        convergence_regions.append(sobol_convergence)

    return convergence_regions



def build_sobol_summary_table(
    region_to_df: dict,
    param_order: list,
    index_type: str = "S1"
):
    """
    Build a summary table of Sobol indices across regions with value ± CI_half and ranking.

    Parameters
    ----------
    region_to_df : dict
        Maps region name → corresponding Si DataFrame (first_Si, total_Si, or second_Si).
    param_order : list
        List of parameter names in the desired row order.
    index_type : str
        One of "S1", "ST", or "S2".

    Returns
    -------
    pd.DataFrame
        Table with one row per parameter, and two columns per region: value ± CI, and rank.
    """
    assert index_type in ["S1", "ST", "S2"]

    rows = []
    regions = list(region_to_df.keys())

    for param in param_order:
        row = {"Parameter": param}
        for region in regions:
            df = region_to_df[region].copy()

            # Ensure there's a 'feature' column
            if "feature" not in df.columns:
                df["feature"] = df.index

            df = df.set_index("feature")

            if param not in df.index:
                row[f"{region}_value"] = "NaN"
                row[f"{region}_rank"] = np.nan
                continue

            value = df.loc[param, index_type]
            conf = df.loc[param, f"{index_type}_conf"]
            formatted = f"{value:.3f} ± {conf:.2f}"
            row[f"{region}_value"] = formatted

        # Rankings
        for region in regions:
            df = region_to_df[region].copy()
            if "feature" not in df.columns:
                df["feature"] = df.index
            df = df.sort_values(by=index_type, ascending=False).reset_index(drop=True)
            rank_map = {name: i + 1 for i, name in enumerate(df["feature"])}
            row[f"{region}_rank"] = rank_map.get(param, np.nan)

        rows.append(row)

    summary_df = pd.DataFrame(rows).set_index("Parameter")

    # Add Sum and Avg Conf if applicable
    if index_type in ["S1", "ST"]:
        sum_row = {"Parameter": "Sum"}
        conf_row = {"Parameter": "Avg Conf"}

        for region in regions:
            df = region_to_df[region]
            sum_val = df[index_type].sum()
            avg_conf = df[f"{index_type}_conf"].mean()
            sum_row[f"{region}_value"] = f"{sum_val:.3f}"
            sum_row[f"{region}_rank"] = ""
            conf_row[f"{region}_value"] = f"{avg_conf:.4f}"
            conf_row[f"{region}_rank"] = ""

        summary_df.loc["Sum"] = sum_row
        summary_df.loc["Avg Conf"] = conf_row

    return summary_df


def select_top_features(
    rank_sources,
    source_type="shap",          # "shap", "S1", or "ST"
    method="threshold",          # "threshold" or "topk"
    threshold=0.90,              # used if method="threshold"
    top_k=6                      # used if method="topk"
):
    """
    Select top features per region based on SHAP or Sobol importance.

    Parameters
    ----------
    rank_sources : dict
        Maps region name → SHAP DataFrame or Sobol Si dict.
    source_type : str
        "shap", "S1", or "ST"
    method : str
        "threshold" (cumulative importance) or "topk" (fixed count)
    threshold : float
        Minimum cumulative importance to reach (only for method="threshold")
    top_k : int
        Number of top features to select (only for method="topk")

    Returns
    -------
    selected_features_per_region : dict
        Region → list of selected features
    union_set : set
        Union of all selected features across regions
    """
    
    if source_type in ["S1", "ST"] and method != "threshold":
        raise ValueError(f"For Sobol source_type '{source_type}', only method='threshold' is supported.")
    
    if source_type in ["shap"] and method != "topk":
        raise ValueError(f"For Shapley FI (source_type = '{source_type}'), only method='topk' is supported.")

    selected_features_per_region = {}

    for region, data in rank_sources.items():
        if source_type == "shap":
            df = data.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
            df["importance"] = df["mean_abs_shap"]
        elif source_type in ["S1", "ST"]:
            df = pd.DataFrame({
                "feature": data["feature"],
                "importance": data[source_type]
            }).sort_values("importance", ascending=False).reset_index(drop=True)
        else:
            raise ValueError("source_type must be 'shap', 'S1', or 'ST'.")

        if method == "threshold":
            df["cumulative"] = df["importance"].cumsum()
            total_importance = df["importance"].sum()
            df["cumulative_norm"] = df["cumulative"] / total_importance

            #return(df)
            # Select features until cumulative_norm >= threshold
            selected = []
            for i, row in df.iterrows():
                selected.append(row["feature"])
                if row["cumulative"] >= threshold:
                    break
            
            total_explained = df[df["feature"].isin(selected)]["importance"].sum()

        elif method == "topk":
            selected = df.head(top_k)["feature"].tolist()
            total_explained = df[df["feature"].isin(selected)]["importance"].sum() / df["importance"].sum()
        else:
            raise ValueError("method must be 'threshold' or 'topk'.")

        selected_features_per_region[region] = selected
        print(f"\nRegion: {region}")
        print(f"Selected {len(selected)} features using method='{method}' "
              f"({f'threshold={threshold}' if method=='threshold' else f'top_k={top_k}'})")
        print(f"Total importance explained: {total_explained:.3f}")
        print("Selected features:", selected)

    # Compute union of all selected features
    union_set = set()
    for features in selected_features_per_region.values():
        union_set.update(features)
 
    print(f"\nUnion of all selected features across regions: {sorted(union_set)}")
    print(f"Total unique features selected: {len(union_set)}")

    return selected_features_per_region, union_set

def build_rank_table(rank_dict):
    """
    Builds a parameter ranking table across regions,
    ordered by the region with the fewest ranked features.

    Parameters
    ----------
    rank_dict : dict
        Dictionary mapping region name to ordered list of ranked features.

    Returns
    -------
    pd.DataFrame
        DataFrame with features as rows and regions as columns.
        Values are rankings (1 = most important), NaN if not ranked.
    """
    # Identify all unique features
    all_features = set(f for lst in rank_dict.values() for f in lst)

    # Use the region with the fewest features to determine row order
    base_region = min(rank_dict, key=lambda k: len(rank_dict[k]))
    base_order = rank_dict[base_region]

    # Append remaining features not in the base region
    remaining = [f for f in all_features if f not in base_order]
    ordered_features = base_order + sorted(remaining)

    # Initialize the DataFrame
    df = pd.DataFrame(index=ordered_features, columns=rank_dict.keys())

    # Fill in rankings
    for region, features in rank_dict.items():
        for rank, feature in enumerate(features, start=1):
            df.loc[feature, region] = rank

    return df.astype("Int64")  # Ensure integer display with NA support


def compare_selected_features(dict_a, dict_b, name_a="SHAP", name_b="Sobol"):
    """
    Compare two feature-selection dictionaries and report:
    - Shared features across all regions (intersection per method)
    - Common features across both methods
    - Unique features added by each method

    Parameters
    ----------
    dict_a : dict
        First selection dictionary (e.g., SHAP).
    dict_b : dict
        Second selection dictionary (e.g., Sobol).
    name_a : str
        Name of first method (for reporting).
    name_b : str
        Name of second method (for reporting).

    Returns
    -------
    common_features : set
        Features shared across all regions and both methods.
    additional_features : dict
        Features unique to each method (not in the shared intersection).
    """
    # Features selected across all regions (intersection within method)
    shared_a = set.intersection(*(set(features) for features in dict_a.values()))
    shared_b = set.intersection(*(set(features) for features in dict_b.values()))

    # Common features (intersection across both methods)
    common_features = shared_a & shared_b

    # Additional features unique to each method
    additional_features = {
        name_a: sorted(shared_a - common_features),
        name_b: sorted(shared_b - common_features)
    }

    return sorted(common_features), additional_features


def add_confidence_intervals(df, index_col="S1", conf_col="S1_conf"):
    """
    Adds confidence interval bounds and a flag if 0 is inside the CI.

    Parameters:
        df (pd.DataFrame): Sobol index DataFrame with index and conf columns.
        index_col (str): Name of the Sobol index column (e.g. 'S1', 'ST', 'S2').
        conf_col (str): Name of the confidence column (e.g. 'S1_conf').

    Returns:
        pd.DataFrame: Updated DataFrame with CI bounds and zero flag.
    """
    df = df.copy()
    df["CI_lower"] = df[index_col] - df[conf_col]
    df["CI_upper"] = df[index_col] + df[conf_col]
    df["CI_contains_0"] = (df["CI_lower"] <= 0) & (df["CI_upper"] >= 0)
    return df


def run_sobol_analysis_for_region(SA, df, aggregation_method,regions=None):
    """
       Run Sobol sensitivity analysis for one or more predefined regions and 
    generate summary statistics for first-, total-, and second-order effects.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataset containing model outputs or AUC values required for the 
        Sobol analysis.
    aggregation_method : str
        Method used to aggregate data before sensitivity analysis (e.g., "AUC").
    regions : dict or None, optional
        Mapping of region names to their corresponding bin indices.
        If None, the configuration is loaded from '../configs/regions_cfg.yaml'.

    Returns
    -------
    results_cmpl : dict
        Dictionary keyed by region name containing:
            - 'Si' : tuple of raw Sobol analysis outputs
            - 'total_Si' : pandas.DataFrame with total-order sensitivity indices (ST)
            - 'first_Si' : pandas.DataFrame with first-order sensitivity indices (S1)
            - 'second_Si' : pandas.DataFrame with second-order sensitivity indices (S2)
    """
    if regions is not None:
        bins = sorted(set(val for sublist in regions.values() for val in sublist))
        _, results_df = SA.run_analysis(df, aggregation_method=aggregation_method, by_regions= True, bins= bins)
    else:
        regions = OmegaConf.load('../configs/regions_cfg.yaml')
        _, results_df = SA.run_analysis(df, aggregation_method=aggregation_method, by_regions= True)
    
    results_cmpl={}
    summary_dict = {}
    for i,region_name in enumerate(regions.keys()):
        # Run sobol
        print(f"[INFO] Running Sobol SA on the AUC of the '{region_name}' region.")
        Si = results_df[i]
        # Convert to DataFrames
        total_Si, first_Si, second_Si = Si

        # Add confidence interval processing
        first_Si = add_confidence_intervals(first_Si, "S1", "S1_conf")
        total_Si = add_confidence_intervals(total_Si, "ST", "ST_conf")
        second_Si = add_confidence_intervals(second_Si, "S2", "S2_conf")

        # Calculate summary metrics
        s1_sum = round(first_Si["S1"].sum(), 4)
        s1_pos_sum = round(first_Si[first_Si["S1"] > 0]["S1"].sum(), 4)

        s2_sum = round(second_Si["S2"].sum(), 4)
        s2_pos_mask = (second_Si["S2"] > 0) & (~second_Si["CI_contains_0"])
        s2_pos_sum = round(second_Si.loc[s2_pos_mask, "S2"].sum(), 4)

        combined_sum = round(s1_sum + s2_sum, 4)
        combined_sig_sum = round(s1_pos_sum + s2_pos_sum, 4)

        if (first_Si["S1"] < 0).any():
            print("[WARNING] Some S1 indices are negative!")
        if (second_Si["S2"] < 0).any():
            print("[WARNING] Some S2 indices are negative!")

        # Store in dict
        summary_dict[region_name] = {
            "Sum of S1 indices": s1_sum,
            "Sum of S1 indices (setting negative indices to 0)": s1_pos_sum,
            "Sum of second order": s2_sum,
            "Sum of second order (only significant & > 0)": s2_pos_sum,
            "Sum of S1 and S2": combined_sum,
            "Sum of significant S1 + significant S2": combined_sig_sum
        }

        
        results_cmpl[region_name] = [Si,total_Si,first_Si,second_Si]
    
    summary_df = pd.DataFrame(summary_dict)
    
    return results_cmpl,summary_df


def create_sobol_sample_dataframe(param_config,parameter_ranges, N=1024, seed=None, calculate_second_order=True):
    """
    Generate a Sobol sampling DataFrame for sensitivity analysis.

    Parameters
    ----------
    param_config : dict
        Parameter configuration including 'name', 'type', 'low', 'high', and optional 'derived'.
    N : int, optional
    seed : int or None, optional
    calculate_second_order : bool, optional
        Compute second-order Sobol indices if True.

    Returns
    -------
    pd.DataFrame
        Sobol samples with discrete parameters adjusted, dependent parameters applied, and an 'index' column.
    """
    
    SA = SensitivityAnalyzer(parameter_ranges, N=N, seed=seed , calculate_second_order=calculate_second_order)
    df = SA.generate_samples()
    
    # Ensure the correct data types for discrete parameters
    discrete_values_dict = {para['name']:list(range(para['low'],(para['high']+1),1)) for para in param_config['parameters'] if para['type']=='integer'}
    if len(discrete_values_dict)> 0:
        for param, valid_values in discrete_values_dict.items():
            df[param] = df[param].apply(lambda x: min(valid_values, key=lambda v: abs(v - x)))
    
    # Define the identifier for the samples
    df_param_values_final = SA.define_id()
    df_param_values_final["index"] = df_param_values_final.index
    ordered_cols = ["config_id", "index"] + df.columns.tolist()
    # Reorder DataFrame
    df_param_values_final = df_param_values_final[ordered_cols]
    return df_param_values_final,SA



def build_feature_ranking_table(rank_dict):
    """
    Create a DataFrame ranking features per region.

    Parameters
    ----------
    feature_dict : dict[str, list[str]]
        Keys are region names; values are ordered features (most important first).
    sort_region : str, optional
        Region to sort rows by. NaNs are placed last.

    Returns
    -------
    pandas.DataFrame
        Features as rows, regions as columns, values are 1-based ranks (NaN if absent).
    """
    
    # all variables across regions (rows)
    all_vars = sorted({v for lst in rank_dict.values() for v in lst})

    # build ranking per region (1-based), then assemble the DataFrame
    df = pd.DataFrame({
        region: pd.Series({var: rank for rank, var in enumerate(vars_, start=1)})
        for region, vars_ in rank_dict.items()
    }).reindex(all_vars).astype('Int64')  # nullable Int to allow NaN for missing

    df = df.sort_values(by='activation')
    return df



