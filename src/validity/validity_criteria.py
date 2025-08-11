# validity/validity_criteria.py

import pandas as pd

def validate_polarization_curves(
    df: pd.DataFrame,
    apply_criteria: dict = {
        "start_in_range": True,
        "approx_monotonic": True,
        "low_ifc_positive_voltage": True
    },
    filter_invalid: bool = False,
    keep_temp_cols: bool = False,
    approx_monotonic_threshold: float = 0.05,
    voltage_range: tuple = (0.0, 1.23),
    early_values_tolerance: int = 3
) -> pd.DataFrame:
    """
    Validate polarization curves in a DataFrame based on customizable criteria.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with polarization curve data (Ucell_1...Ucell_N, ifc_1...ifc_N)

    apply_criteria : dict
        Dictionary of criteria to apply:
            - "start_in_range"
            - "early_values_in_range"
            - "monotonic"
            - "approx_monotonic"
            - "low_ifc_positive_voltage"

    filter_invalid : bool
        If True, removes rows classified as 'invalid'

    keep_temp_cols : bool
        If False, removes temporary boolean columns

    approx_monotonic_threshold : float
        Threshold for approximate monotonicity

    voltage_range : tuple
        Min and max acceptable voltages

    early_values_tolerance : int
        Number of early Ucell values to validate as in range

    Returns
    -------
    pd.DataFrame
        DataFrame with classification and optionally filtered
    """

    df_config = df.copy()
    ucell_columns = [col for col in df.columns if col.startswith("Ucell_")]
    ifc_columns = [col for col in df.columns if col.startswith("ifc_")]
    v_min, v_max = voltage_range

    # Criterion: first Ucell value within voltage range
    if apply_criteria.get("start_in_range", False):
        df_config["start_in_range"] = df_config[ucell_columns[0]].between(v_min, v_max)

    # Criterion: early Ucell values within range
    if apply_criteria.get("early_values_in_range", False):
        df_config["early_values_in_range"] = df_config[ucell_columns[:early_values_tolerance]].apply(
            lambda row: row.between(v_min, v_max).all(), axis=1
        )

    # Criterion: strictly non-increasing
    if apply_criteria.get("monotonic", False):
        df_config["monotonic"] = df_config[ucell_columns].apply(
            lambda row: all(x >= y for x, y in zip(row, row[1:])), axis=1
        )

    # Criterion: approximately non-increasing
    if apply_criteria.get("approx_monotonic", False):
        def approx_monotonic(row, threshold=approx_monotonic_threshold):
            voltages = row.values.astype(float)
            for i in range(len(voltages) - 1):
                if voltages[i] < voltages[i + 1] - threshold:
                    return False
            return True

        df_config["approx_monotonic"] = df_config[ucell_columns].apply(approx_monotonic, axis=1)

    # New Criterion: ifc < 2 => ucell must be > 0
    if apply_criteria.get("low_ifc_positive_voltage", False):
        def low_ifc_positive_voltage(row):
            for i in range(len(ifc_columns)):
                if row[ifc_columns[i]] < 2.0 and row[ucell_columns[i]] <= 0:
                    return False
            return True

        df_config["low_ifc_positive_voltage"] = df_config.apply(low_ifc_positive_voltage, axis=1)

    # Combine active criteria into one final classification column
    criteria_cols = [key for key in [
        "start_in_range",
        "early_values_in_range",
        "monotonic",
        "approx_monotonic",
        "low_ifc_positive_voltage"
    ] if apply_criteria.get(key, False)]

    df_config["classification"] = df_config[criteria_cols].all(axis=1)
    df_config["classification"] = df_config["classification"].map({True: "valid", False: "invalid"})

    if filter_invalid:
        df_config = df_config[df_config["classification"] == "valid"]
        df_config = df_config.drop(columns=["classification"])

    if not keep_temp_cols:
        df_config = df_config.drop(columns=criteria_cols, errors="ignore")

    return df_config

