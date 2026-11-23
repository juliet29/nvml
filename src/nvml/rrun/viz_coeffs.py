import numpy as np
import polars as pl
from matplotlib.axes import Axes

from nvml.rrun.interfaces import CoeffData as cd


def plot_coeffs(df: pl.DataFrame, defn: str, ax: Axes):
    assert defn in df[cd.defn].unique().to_list()
    cdf = df.filter(pl.col(cd.defn) == defn).filter(pl.col(cd.feature) != cd.intercept)
    if cdf.is_empty():
        raise ValueError(f"df is empty after filtering..{df}")
    coeffs = cdf[cd.coeff].to_numpy()
    lower_errors = coeffs - cdf[cd.ci_lower].to_numpy()
    upper_errors = cdf[cd.ci_upper].to_numpy() - coeffs

    ax.errorbar(
        x=coeffs,
        y=cdf[cd.feature].to_list(),
        xerr=np.vstack([lower_errors, upper_errors]),
        fmt="o",
        capsize=4,
    )
    ax.axvline(0, color="gray", linestyle="--", linewidth=1)
    ax.invert_yaxis()
    return ax

    # need to look at distribution of area to scale it
    #
    # at some point need to standardize, and do exponential scaling..

    # point
    #
    # erro bars
