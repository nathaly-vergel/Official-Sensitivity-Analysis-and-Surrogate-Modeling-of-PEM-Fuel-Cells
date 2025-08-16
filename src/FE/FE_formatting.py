import os
import sys
import hashlib
import pickle
from omegaconf import OmegaConf
# For data science in general
import numpy as np
import pandas as pd


def expand_column_to_columns(df_or, param='ifc'):
    """
    Expands a column in a DataFrame containing iterable elements into separate columns.

    Parameters
    ----------
    df_or : pandas.DataFrame
        The input DataFrame that contains the column to be expanded.
    param : str, optional, default='ifc'
        The name of the column in df_or to expand. Each element in this column
        should be iterable (like a list or numpy array) of the same length.

    Returns
    -------
    pandas.DataFrame
        A new DataFrame with the specified column expanded into multiple columns.
        The new columns are named as '{param}_0', '{param}_1', ..., '{param}_{n-1}',
        where n is the length of the iterable in the first row of the specified column.
        The original column is preserved in the returned DataFrame.

    Notes
    -----
    - If any element in the column is None, it will be replaced with an array of NaNs.
    """
    df = df_or.copy()
    len_param = len(df.iloc[0][param])
    param_nm = param + '_copy'
    df[param_nm] = df[param]
    df[param_nm] = df[param_nm].apply(lambda x: x if x is not None else np.full(len_param, np.nan))
    df_expanded = pd.DataFrame(df[param_nm].tolist(), columns=[f'{param}_{i}' for i in range(len_param)])
    df_merged_large = pd.concat([df.drop(columns=[param_nm]).reset_index(drop=True), df_expanded], axis=1)
    return df_merged_large

def parse_dependent_parameters (dependent_list):
    """
    Convert derived parameter definitions into callable functions.

    Each dict in `dependent_list` must have:
        - 'name': derived parameter name
        - 'derived': expression like "Pa_des - 20000"

    Supported operators: '+', '-', '*', '/'

    Returns a list of dicts with:
        - 'parameter_name': name of the derived parameter
        - 'function': lambda implementing the derivation
        - 'dependent_param': name of the dependent parameter

    Raises ValueError if an unsupported operator is used.
    """
    new_list = []
    for p in dependent_list:
        dep_param, op, val = p['derived'].split()
        val = float(val) if '.' in val else int(val)

        if op == '-':
            func = lambda x, v=val: x - v
        elif op == '+':
            func = lambda x, v=val: x + v
        elif op == '*':
            func = lambda x, v=val: x * v
        elif op == '/':
            func = lambda x, v=val: x / v
        else:
            raise ValueError(f"Unknown operator {op}")

        new_list.append({
            'parameter_name': p['name'],
            'function': func,
            'dependent_param': dep_param
        })
    return new_list


def load_parameter_ranges(path_param_config):
    """
    Load a parameter configuration and extract ranges for non-fixed parameters.

    Parameters
    ----------
    path_param_config : str
        Path to the YAML parameter configuration file.

    Returns
    -------
    param_config : dict or DictConfig
        Full loaded parameter configuration.
    parameter_ranges : dict
        Mapping of parameter names to [low, high] values for non-fixed parameters.
    parameter_names : list
        List of non-fixed parameter names.
    """
    param_config = OmegaConf.load(path_param_config)
    parameter_ranges = {para['name']: [para['low'], para['high']] 
                for para in param_config['parameters'] if para['fixed']==False}
    parameter_names = list(parameter_ranges.keys())
    return param_config,parameter_ranges, parameter_names