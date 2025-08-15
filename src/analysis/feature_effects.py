import sys
import os 
import random
import numpy as np
import pandas as pd
import shap
import random
import xgboost as xgb
from omegaconf import OmegaConf
from SALib.sample import sobol
from SALib.analyze import sobol as sobol_analyze
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
from src.analysis.sensitivity import SensitivityAnalyzer

REGIONS = OmegaConf.load('../configs/regions_cfg.yaml')
core_training = OmegaConf.load('../configs/core_training_cfg.yaml')

class FeatureEffects:
    def __init__(self, model, parameter_ranges, region=None, seed=None):
        if region is not None and region not in REGIONS:
            raise ValueError(
                f"Invalid region: '{region}'. Must be one of: {list(REGIONS.keys())}"
            )
        self.model = model
        self.region = region if region is not None else REGIONS
        self.parameter_ranges = parameter_ranges
        self.seed = seed if seed is not None else core_training['seed']
        if region is not None:
            self.parameter_ranges['ifc'] = REGIONS[region]


    def ensure_numeric_dataframe(self, df):
        """
        Ensure that all columns in a DataFrame are numeric (float or int).
        Converts object columns to float if possible.
        Raises an error if conversion fails.

        Parameters:
        - X: pd.DataFrame

        Returns:
        - pd.DataFrame with only numeric columns
        """
        X_checked = df.copy()
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
    
    def sample_region(self, df):
        """
        Samples a region from the DataFrame based on the specified region bounds.

        Parameters
        ----------
        df : pd.DataFrame
            The input DataFrame containing feature data.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing only the rows that fall within the specified region.
        """
        if self.region is None:
            return df

        lower_bound, upper_bound = REGIONS[self.region]
        sampled_df = df[(df['ifc'] >= lower_bound) & (df['ifc'] < upper_bound)]
        
        if sampled_df.empty:
            raise ValueError(f"[ERROR] No data found in the specified region '{self.region}'.")

        return sampled_df

    def shap(self,df, target_col, sample_frac=1.0, verbose=False):
        """
        Computes SHAP feature importance for a trained XGBoost model on AUC prediction.

        Parameters
        ----------
        target_col : str
            Name of the target column in the dataset.
        sample_frac : float, optional
            Fraction of data to sample for SHAP computation (default 1.0).
        verbose : bool, optional
            Whether to print progress info.

        Returns
        -------
        pd.DataFrame
            SHAP feature importance ranking with columns ['feature', 'mean_abs_shap'].
        """
        np.random.seed(self.seed)
        random.seed(self.seed)
        input_cols = list(self.parameter_ranges) + ([] if 'ifc' in self.parameter_ranges else ['ifc'])

        if target_col not in df.columns:
            raise ValueError(f"[ERROR] Column '{target_col}' not found in the dataframe.")

        df_region = df.dropna(subset=[target_col]).copy()
        X_full = df_region[input_cols]
        X_full = self.sample_region(X_full)
        X_full = self.ensure_numeric_dataframe(X_full)

        if sample_frac < 1.0:
            X_sample = X_full.sample(frac=sample_frac, random_state=self.seed)
            if verbose:
                print(f"[INFO] Using a sample of {len(X_sample)} rows from {len(X_full)} total rows for SHAP.")
        else:
            X_sample = X_full
            if verbose:
                print(f"[INFO] Using full dataset ({len(X_sample)} rows) for SHAP.")

        if verbose:
            print("[INFO] Creating SHAP TreeExplainer...")
        explainer = shap.Explainer(self.model)

        if verbose:
            print("[INFO] Computing SHAP values...")
            print(f"[INFO] columns {X_sample.columns.to_list()}")
        shap_values = explainer(X_sample)

        if verbose:
            print("[INFO] Aggregating mean absolute SHAP values per feature...")
        mean_abs_shap = pd.DataFrame({
            "feature": X_sample.columns,
            "mean_abs_shap": np.abs(shap_values.values).mean(axis=0)
        }).sort_values(by="mean_abs_shap", ascending=False)

        return shap_values, mean_abs_shap
    

    def sobol(self, N_list, calculate_second_order=False,verbose=False,ifc=True):
        np.random.seed(self.seed)
        random.seed(self.seed)
        results_cmpl = {}

        if ifc:
            self.parameter_ranges['ifc'] = REGIONS[self.region]

        for N in N_list:
            SA = SensitivityAnalyzer(
                self.parameter_ranges,
                dependent_parameter_names=None,
                seed=self.seed,
                N=N,
                calculate_second_order=calculate_second_order)
            
            df_X = SA.generate_samples()
            
            #samples = SA.define_id()
            input_cols = list(self.parameter_ranges.keys())
            df_X['Ucell'] = self.model.predict(df_X)
            
            results,_ = SA.run_analysis(df_X, aggregation_method=None, by_regions= False)
            results = results[0]
            
            if verbose:
                print(f"[INFO] Analyzing {N} samples...")

            S1 = np.array(results["S1"])
            S1_conf = np.array(results["S1_conf"])
            ST = np.array(results["ST"])
            ST_conf = np.array(results["ST_conf"])

            S1_CI = [tuple(np.round([low, high], 3)) for low, high in zip(S1 - S1_conf, S1 + S1_conf)]
            ST_CI = [tuple(np.round([low, high], 3)) for low, high in zip(ST - ST_conf, ST + ST_conf)]


            if verbose:
                print(f"[INFO] S1")

            data = {
                "feature": input_cols,
                "S1": S1,
                "S1_CI": S1_CI,
                "ST": ST,
                "ST_CI": ST_CI
            }
            
            s2_sum = 0.0
            if calculate_second_order:
                if verbose:
                    print("[INFO] Calculating second-order Sobol indices...")
                S2 = np.array(results["S2"])
                S2_conf = np.array(results["S2_conf"])

                # Row-wise sum of second-order interactions and corresponding CI half-widths
                S2_total = np.nansum(S2, axis=1)
                S2_total_conf = np.nansum(S2_conf, axis=1)
                S2_CI = [tuple(np.round([low, high], 3)) for low, high in zip(S2_total - S2_total_conf, S2_total + S2_total_conf)]

                data["S2_total"] = S2_total
                data["S2_CI"] = S2_CI

                s2_sum = np.nansum(np.triu(S2, k=1))  # only upper triangle (unique pairs)
            if verbose:
                print(f"[INFO] final df data")
            sobol_df = pd.DataFrame(data).sort_values("ST", ascending=False).reset_index(drop=True)

            # Diagnostics
            s1_sum = np.nansum(S1)
            st_sum = np.nansum(ST)
            if verbose:
                print(f"[INFO] Sum S1 = {s1_sum:.4f}")
                if calculate_second_order:
                    print(f"[INFO] Sum S2 (unique pairs) = {s2_sum:.4f}")
                    print(f"[INFO] Sum S1 + S2 = {s1_sum + s2_sum:.4f} (should be ≈ 1)")
                print(f"[INFO] Sum ST = {st_sum:.4f} (may be > 1 due to overlaps)")

                if calculate_second_order and not np.isclose(s1_sum + s2_sum, 1.0, atol=0.1):
                    print("[WARNING] S1 + S2 does not sum to 1 — surrogate may be biased or incomplete.")

            # Store both processed and raw results
            results_cmpl[N] = {
                "sobol_df": sobol_df,
                "Si": results,
                "s1_sum": s1_sum,
                "s2_sum": s2_sum,
                "st_sum": st_sum
            }

        return results_cmpl
                






