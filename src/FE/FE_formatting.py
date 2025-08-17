import os
import sys
import hashlib
import pickle
from omegaconf import OmegaConf
# For data science in general
import numpy as np
import pandas as pd


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