# src/visualization/polarization.py
from __future__ import annotations

from typing import Optional, Sequence, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


__all__ = ["plot_polarization_curves"]


def _extract_curve_from_row(row: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract (ifc, Ucell) arrays from a row that has expanded columns "ifc_1.."
    and "Ucell_1..". Drops NaNs and sorts by current density.
    """
    ucell_cols = [c for c in row.index if str(c).startswith("Ucell_")]
    ifc_cols   = [c for c in row.index if str(c).startswith("ifc_")]

    if not ucell_cols or not ifc_cols:
        raise ValueError("Row does not contain expanded 'Ucell_*' and 'ifc_*' columns.")

    voltages = row[ucell_cols].astype(float).to_numpy()
    currents = row[ifc_cols].astype(float).to_numpy()

    # Keep only finite, paired points
    mask = np.isfinite(currents) & np.isfinite(voltages)
    currents = currents[mask]
    voltages = voltages[mask]

    # Sort by current density for a clean curve
    order = np.argsort(currents)
    return currents[order], voltages[order]


def plot_polarization_curves(
    rows: pd.DataFrame,
    *,
    print_legend: bool = True,
    label_mode: str = "auto",         # "auto" | "index" | "config_id" | "none"
    max_legend_items: int = 15,
    id_column_candidates: Sequence[str] = ("config_id", "id"),
    title: str = "Polarization curve",
    xlabel: str = r"Current density $i_{fc}$ (A·cm$^{-2}$)",
    ylabel: str = r"Cell voltage $U_{cell}$ (V)",
    figsize: Tuple[float, float] = (8, 5),
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot one or more polarization curves from a DataFrame that contains the expanded
    columns `ifc_1..ifc_N` and `Ucell_1..Ucell_N` (as produced by your pipeline).

    Parameters
    ----------
    rows : pd.DataFrame
        A DataFrame slice (or whole DF) where each row represents a configuration and
        includes expanded columns `ifc_*` and `Ucell_*`. It should also carry either
        an index or an identifier column like `config_id` for labeling.
    print_legend : bool, default True
        Whether to show a legend.
    label_mode : {"auto","index","config_id","none"}, default "auto"
        - "auto": if `config_id` (or `id`) exists and there are <= `max_legend_items`,
          label with that; otherwise use the row index if <= `max_legend_items`;
          otherwise no legend labels.
        - "index": force labels to be the DataFrame index (if <= max_legend_items).
        - "config_id": force labels from `config_id` (or first available in
          `id_column_candidates`) if present and <= max_legend_items.
        - "none": no labels.
    max_legend_items : int, default 15
        Only show legend labels if the number of curves is <= this threshold.
    id_column_candidates : sequence of str, default ("config_id","id")
        Column names to try (in order) when label_mode wants a config id.
    title, xlabel, ylabel : str
        Plot titles and axis labels.
    figsize : (float, float)
        Figure size in inches.

    Returns
    -------
    (fig, ax) : tuple
        Matplotlib Figure and Axes for further customization.

    Notes
    -----
    * Each curve uses Matplotlib's default color cycle (distinct colors per curve).
    * Rows with missing/NaN points are handled gracefully; trailing NaNs from padding
      are dropped automatically.
    * Current-density points are sorted before plotting.
    """
    if not isinstance(rows, pd.DataFrame):
        raise TypeError("`rows` must be a pandas DataFrame (e.g., df.loc[mask]).")

    n_curves = len(rows)
    fig, ax = plt.subplots(figsize=figsize)

    # Decide labeling strategy
    def pick_id_column(df: pd.DataFrame) -> Optional[str]:
        for col in id_column_candidates:
            if col in df.columns:
                return col
        return None

    id_col = pick_id_column(rows)

    def get_label(row: pd.Series, idx_label: str) -> Optional[str]:
        # idx_label is stringified index for stability in legend
        if label_mode == "none":
            return None

        if label_mode == "config_id":
            if id_col and n_curves <= max_legend_items:
                return f"{rows.at[row.name, id_col]}"
            return None

        if label_mode == "index":
            return idx_label if n_curves <= max_legend_items else None

        # auto mode
        if id_col and n_curves <= max_legend_items:
            return f"{rows.at[row.name, id_col]}"
        if n_curves <= max_legend_items:
            return idx_label
        return None

    # Plot
    for idx, (_, row) in enumerate(rows.iterrows(), start=1):
        x, y = _extract_curve_from_row(row)
        label = get_label(row, idx_label=str(row.name))
        ax.plot(x, y, marker="o", label=label)

    ax.axhline(0.0, linestyle="--", linewidth=1.0)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True)

    # Legend
    if print_legend:
        # Only show legend entries that actually have labels
        handles, labels = ax.get_legend_handles_labels()
        if labels:
            ax.legend(loc="best")

    fig.tight_layout()
