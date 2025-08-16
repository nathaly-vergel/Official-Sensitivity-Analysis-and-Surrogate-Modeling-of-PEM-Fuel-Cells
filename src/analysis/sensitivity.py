import random
import numpy as np
import pandas as pd
import hashlib
from omegaconf import OmegaConf
import matplotlib.pyplot as plt
from SALib.sample import sobol
from SALib.analyze import sobol as sobol_analyze
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="SALib")



class SensitivityAnalyzer:
    def __init__(self, parameter_ranges, dependent_parameter_names=None, 
                 seed=42, N=10, num_levels=4, calculate_second_order=False):
        self.parameter_ranges = parameter_ranges
        self.dependent_parameter_names = dependent_parameter_names or []
        self.seed = seed
        self.N = N
        self.num_levels = num_levels
        self.problem = self._define_problem()
        self.samples_df = None
        self.calculate_second_order = calculate_second_order

    def _define_problem(self):
        discrete_values_dict = {}
        processed_bounds = {}
        param_dict = self.parameter_ranges 
        for param, values in param_dict.items():
            if len(values) > 2:
                discrete_values_dict[param] = values
                processed_bounds[param] = [min(values), max(values)]
            else:
                # Continuous: Use as-is
                processed_bounds[param] = values
        param_names = list(processed_bounds.keys())
        bounds = [processed_bounds[name] for name in param_names]

        problem = {
            'num_vars': len(param_names),
            'names': param_names,
            'bounds': bounds
        }
        return problem


    def generate_samples(self):
        np.random.seed(self.seed)
        random.seed(self.seed)
        samples = sobol.sample(self.problem, N=self.N, calc_second_order=self.calculate_second_order, seed=self.seed)
        self.samples_df = pd.DataFrame(samples, columns=self.problem["names"])
        return self.samples_df

    def define_id(self):
        df = self.samples_df.copy()
        def generate_row_id(row):
            # Convert all values to string, concatenate, and hash
            row_string = '|'.join([str(x) for x in row])
            return hashlib.sha256(row_string.encode()).hexdigest()
        df['config_id'] = df.apply(generate_row_id, axis=1)
        self.samples_df = df
        return df

    def apply_dependent_parameters(self, dependent_parameters):
        np.random.seed(self.seed)
        df = self.samples_df.copy()
        for param in dependent_parameters:
            new_col = param['parameter_name']
            func = param['function']
            dep = param['dependent_param']
            df[new_col] = df.apply(lambda row: func(row[dep]), axis=1)
        self.samples_df = df
        return df
    
    def aggregate_output_function(self, data, aggregation_method, by_regions=False, bins=None):
        

        def is_valid_array(arr):
            return isinstance(arr, (list, np.ndarray)) and not pd.isna(arr).all()

        if by_regions:
            # --- Validate `bins` ---
            if bins is None:
                regions = OmegaConf.load('../configs/regions_cfg.yaml')
                bins = [v[0] for v in list(regions.values())] + [list(regions.values())[-1][1]]
            else:
                if not isinstance(bins, (list, tuple, np.ndarray)):
                    raise TypeError("`bins` must be a list, tuple, or numpy array.")
                if len(bins) != 4:
                    raise ValueError("`bins` must contain exactly 4 numeric values to define 3 regions.")
                if not all(isinstance(b, (int, float, np.integer, np.floating)) for b in bins):
                    raise TypeError("All elements in `bins` must be numeric.")
                if not all(bins[i] < bins[i + 1] for i in range(len(bins) - 1)):
                    raise ValueError("`bins` must be strictly increasing.")

            labels = ['activation', 'ohmic', 'mass']

            def process_row(row):
                ucell, ifc = row['Ucell'], row['ifc']

                if not (is_valid_array(ucell) and is_valid_array(ifc)):
                    return [np.nan] * len(labels)

                ucell = np.array(ucell)
                ifc = np.array(ifc)

                if len(ucell) != len(ifc):
                    return [np.nan] * len(labels)

                # Group data into regions using bins
                grouped = pd.cut(ifc, bins=bins, labels=labels, right=False)

                result = []
                for label in labels:
                    region_idx = np.where(grouped == label)[0]
                    if len(region_idx) == 0:
                        result.append(np.nan)
                    elif aggregation_method == "sum":
                        result.append(np.sum(ucell[region_idx]))
                    elif aggregation_method == "AUC":
                        if len(region_idx) < 2:
                            result.append(0.0)
                        else:
                            result.append(np.trapezoid(x=ifc[region_idx], y=ucell[region_idx]))
                    else:
                        raise ValueError(f"Unsupported aggregation method: {aggregation_method}")

                return result

            if aggregation_method == "fPCA":
                raise ValueError("fPCA method is not compatible with region-based aggregation. Please use 'sum' or 'AUC'.")

            result = data.apply(process_row, axis=1)
            return result

        else:
            if aggregation_method == "sum":
                return data['Ucell'].apply(lambda x: np.sum(x))

            elif aggregation_method == "AUC":
                return data.apply(lambda row: np.trapezoid(x=row['ifc'],y=row['Ucell']) if row['Ucell'] is not None and row['ifc'] is not None else np.nan, axis=1)

            elif aggregation_method == "fPCA":
                valid_ucell = data['Ucell'].apply(lambda x: x is not None and isinstance(x, (list, np.ndarray)))
                filtered_ucell = data.loc[valid_ucell, 'Ucell']
                Ucell_matrix = np.stack(filtered_ucell.apply(np.array))
                n_components = 5
                pca = PCA(n_components=n_components)
                scores = pca.fit_transform(Ucell_matrix)
                weights = pca.explained_variance_ratio_[:n_components]
                scalar_outputs = (scores[:, :n_components] * weights).sum(axis=1)
                return scalar_outputs
            else:
                raise ValueError(f"Unknown aggregation method: {aggregation_method}")
        
            
        
    def run_analysis(self, data, aggregation_method, by_regions=False,bins=None):
        if aggregation_method is None:
            outputs = data['Ucell']
        elif by_regions:
            outputs = self.aggregate_output_function(data, aggregation_method, by_regions, bins=bins)
        else:
            outputs = self.aggregate_output_function(data, aggregation_method)

        outputs = np.stack(outputs)
        if outputs.ndim == 1:
            outputs = outputs[:, np.newaxis]
        n_outputs = outputs.shape[1]
        results = []
        results_df = []
        for i in range(n_outputs):
            analysis = sobol_analyze.analyze(
                problem=self.problem,
                Y=outputs[:, i],
                print_to_console=False,
                calc_second_order=self.calculate_second_order
            )
            # first‐order and total‐order (as before)
            ST, S1, S2 = analysis.to_df()
            entry = {
                'param': self.problem['names'],
                'output_index': i,
                'S1': analysis['S1'],
                'S1_conf': analysis['S1_conf'],
                'ST': analysis['ST'],
                'ST_conf': analysis['ST_conf'],
            }
            # add second‐order if available
            if self.calculate_second_order:
                # S2 is a D×D symmetric matrix; we can store it as-is, 
                # or flatten only the upper triangle, etc.
                entry['S2'] = analysis['S2']                   # full matrix
                entry['S2_conf'] = analysis['S2_conf']        # same shape
            results.append(entry)
            results_df.append([ST, S1, S2])

        return results,results_df
    
    
        