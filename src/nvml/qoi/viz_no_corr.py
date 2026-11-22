import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns
import xarray as xr
from polars import selectors as cs

from nvml import dn


def annotate_corr(x, y, **kwargs):
    r = x.corr(y)

    ax = plt.gca()
    ax.annotate(
        f"r = {r:.2f}",
        xy=(0.05, 0.85),  # top left of each axis
        xycoords=ax.transAxes,
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
    )


def plot_no_qoi_corr(flow_qoi: xr.Dataset):
    # these should be the wind binned values
    wind_bin = flow_qoi.groupby(dn.wind_sector).median()
    pdf = wind_bin.to_dataframe().reset_index()

    g = sns.PairGrid(
        pdf,
        vars=[
            dn.intake,
            dn.dist,
            dn.cv,
        ],  # The numeric columns to pair against each other
        hue=dn.wind_sector,  # Categorical variable for color grouping
        corner=True,
    )
    g.map_lower(sns.scatterplot, alpha=0.25, s=12, edgecolor="none")
    g.map_diag(sns.kdeplot, fill=True)
    g.add_legend()

    corr = pl.DataFrame(pdf).select(cs.float()).corr()

    return g.figure, corr
