import os
import sys
import hashlib
import pickle

# For data science in general
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances


def compute_nearest_neighbors(df, parameter_ranges, k=10):
    """
    Calculates the k-nearest neighbors of "no output" cases based on valid cases in a DataFrame.

    This function selects rows from the DataFrame classified as "valid" and "invalid_no_output" (with missing 'Ucell'),
    scales the input parameters, computes pairwise Euclidean distances between "no output" and valid cases, 
    and identifies the k closest neighbors for each "no output" sample.

    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame containing data points, classification labels, and input parameters.
    k : int, optional, default=10
        The number of nearest neighbors to identify for each "no output" sample.

    Returns
    -------
    nearest_indices : numpy.ndarray
        Array of shape (num_no_output_samples, k) containing the indices of the k closest valid neighbors
        for each "no output" case.
    nearest_distances : numpy.ndarray
        Array of shape (num_no_output_samples, k) containing the Euclidean distances to the k closest valid neighbors
        for each "no output" case.
    final_df_valid_cmpl : pandas.DataFrame
        Subset of the original DataFrame containing rows classified as "valid".
    final_df_no_output_cmpl : pandas.DataFrame
        Subset of the original DataFrame containing rows classified as "invalid_no_output" 
        and having missing 'Ucell'.

    
    """
    # Select only rows classified as "valid" from the validated DataFrame
    final_df_valid_cmpl = df.loc[df.classification == "valid"]
    final_df_valid = final_df_valid_cmpl[parameter_ranges.keys()]

    # Select rows classified as "invalid_no_output" and where 'Ucell' is missing (NaN)
    final_df_no_output_cmpl = df.loc[(df.classification == "invalid_no_output") & (df['Ucell'].isna())]

    # Keep only the columns corresponding to the input parameters for the "no output" cases
    final_df_no_output = final_df_no_output_cmpl[parameter_ranges.keys()]

    # Print the shapes of the resulting DataFrames for quick inspection
    print(f'Shape of valid cases: {final_df_valid.shape} & shape of no-output cases: {final_df_no_output.shape}')

    # Combine the valid and "no output" DataFrames into a single DataFrame
    combined_df = pd.concat([final_df_valid, final_df_no_output], ignore_index=True)

    scaler = StandardScaler()
    combined_scaled = scaler.fit_transform(combined_df)

    # Split the scaled data back into two sets:
    # - query_scaled: the first len(final_df_valid) rows (the "valid" cases)
    # - database_scaled: the remaining rows (the "no output" cases)
    query_scaled = combined_scaled[:len(final_df_valid)]
    database_scaled = combined_scaled[len(final_df_valid):]

    # Compute the pairwise Euclidean distance matrix between database and query sets
    # Each row corresponds to a "no output" case, each column corresponds to a "valid" case
    distance_matrix = pairwise_distances(database_scaled, query_scaled, metric='euclidean')

    print(f'Shape of distance matrix: {distance_matrix.shape}')

    # Identify the indices of the k closest neighbors for each "no output" sample
    nearest_indices = np.argsort(distance_matrix, axis=1)[:, :k]

    # Identify the distances to the k closest neighbors for each "no output" sample
    nearest_distances = np.sort(distance_matrix, axis=1)[:, :k]

    return nearest_indices, nearest_distances, final_df_valid_cmpl, final_df_no_output_cmpl

def aggregate_knn_curves(df_valid,df_missing,nearest_indices,agg_fn='mean'):
    """
    Computes aggregated curves for missing data points using their nearest valid neighbors.

    For each row in `df_missing`, this function selects the corresponding nearest neighbors
    from `df_valid` (indices provided in `nearest_indices`) and aggregates their `ifc` and
    `Ucell` curves using the specified aggregation function (`mean` or `median`). It returns
    a new DataFrame containing the averaged curves along with identifiers.

    Parameters
    ----------
    df_valid : pandas.DataFrame
        DataFrame containing valid curves, must include columns 'ifc' and 'Ucell'.
    df_missing : pandas.DataFrame
        DataFrame containing rows with missing outputs.
    nearest_indices : array-like
        Array of shape (num_missing_rows, k) with indices of the nearest valid neighbors
        for each missing row.
    agg_fn : str, optional, default='mean'
        Aggregation function to use for Ucell curves. Options: 'mean' or 'median'.

    Returns
    -------
    df_knn_missing : pandas.DataFrame
        DataFrame containing the identifiers of missing rows along with the aggregated
        'ifc' and 'Ucell' curves.
    ifc_outputs_knn : list of numpy.ndarray
        List containing the aggregated 'ifc' curves for each missing row.
    ucell_outputs_knn : list of numpy.ndarray
        List containing the aggregated 'Ucell' curves for each missing row.
    """
    ucell_outputs_knn=[]
    ifc_outputs_knn=[]
    for i, neighbors in enumerate(nearest_indices):
        ifc_mean = np.stack(df_valid.iloc[neighbors]['ifc']).mean(axis=0)
        ifc_outputs_knn.append(ifc_mean)
        if agg_fn == 'mean':
            ucell_mean = np.stack(df_valid.iloc[neighbors]['Ucell']).mean(axis=0)
            ucell_outputs_knn.append(ucell_mean)
        elif agg_fn == 'median':
            ucell_median = np.median(np.stack(df_valid.iloc[neighbors]['Ucell']), axis=0)
            ucell_outputs_knn.append(ucell_median)
        else:
            raise ValueError("agg_fn must be either 'mean' or 'median'")
    
    df_knn_missing = pd.concat([df_missing[['config_id', 'index']].reset_index(drop=True),
    pd.DataFrame(np.stack(ifc_outputs_knn), columns=[f'ifc_{i}' for i in range(len(ifc_outputs_knn[0]))]),
    pd.DataFrame(np.stack(ucell_outputs_knn), columns=[f'Ucell_{i}' for i in range(len(ucell_outputs_knn[0]))])], axis=1)
    
    return df_knn_missing,ifc_outputs_knn,ucell_outputs_knn