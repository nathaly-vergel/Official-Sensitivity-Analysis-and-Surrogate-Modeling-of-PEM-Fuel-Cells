import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def plot_residuals_vs_ifc(ifc, residuals, figsize=(5, 3), alpha=0.01, s=10, title="Residuals vs ifc"):
    """
    Scatter plot of residuals vs ifc with a horizontal line at 0.

    Args:
        ifc (array-like): The x-values (ifc).
        residuals (array-like): Residual values (y-values).
        figsize (tuple): Figure size.
        alpha (float): Transparency of scatter points.
        s (int): Size of scatter markers.
        title (str): Plot title.
    """
    plt.figure(figsize=figsize)
    sns.scatterplot(x=ifc, y=residuals, alpha=alpha, s=s)
    plt.axhline(0, color="red", linestyle="--")
    plt.title(title)
    plt.xlabel("Current Density [A/cm²]")
    plt.ylabel("Residuals")
    plt.tight_layout()
    plt.show()

def plot_residuals_by_ifc_boxplots(test_df, residuals, n_bins=30, figsize=(14,6), color='skyblue', title="Residual Distribution by Current Density [A/cm$^2$]"):
    """
    Plot residuals binned by current density (ifc) as a boxplot.

    Args:
        test_df (pd.DataFrame): DataFrame containing 'ifc' column.
        residuals (array-like): Residuals corresponding to test_df rows.
        n_bins (int): Number of bins to group 'ifc' values.
        figsize (tuple): Figure size for the plot.
        color (str): Boxplot color.
        title (str): Plot title.
    """
    df = test_df.copy()
    df['ifc_bin'] = pd.cut(df['ifc'], bins=n_bins, labels=False) + 1  # 1-indexed bins
    df['residual'] = residuals

    # Calculate bin centers for x-axis labels
    bin_edges = pd.cut(df['ifc'], bins=n_bins).unique().categories
    bin_centers = [(interval.left + interval.right) / 2 for interval in bin_edges]
    bin_labels = [f"{center:.2f}" for center in bin_centers]

    plt.figure(figsize=figsize)
    sns.boxplot(
        data=df,
        x='ifc_bin',
        y='residual',
        color=color,
        fliersize=2,
        linewidth=1
    )

    plt.axhline(0, color="red", linestyle="--")
    plt.xticks(ticks=range(len(bin_labels)), labels=bin_labels, fontsize=9)
    plt.title(title, fontsize=17)
    plt.xlabel("Current Density [A/cm²]", fontsize=17)
    plt.ylabel("Residuals", fontsize=17)
    plt.grid(True, linestyle='--', axis='y', alpha=0.5)
    plt.tight_layout()
    plt.show()

def plot_residuals_hist(residuals, bins=50, figsize=(8,5), title="Histogram of Residuals"):
    """
    Plot a histogram of residuals.

    Args:
        residuals (array-like): Residuals to plot.
        bins (int): Number of histogram bins.
        figsize (tuple): Figure size.
        title (str): Plot title.
    """
    plt.figure(figsize=figsize)
    plt.hist(residuals, bins=bins, edgecolor='black', alpha=0.7)
    plt.title(title)
    plt.xlabel("Residual")
    plt.ylabel("Frequency")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()
