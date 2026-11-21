import matplotlib.pyplot as plt
import seaborn as sns
import xarray as xr

from nvml import dn


def plot_ambient_dep(qoi: str, flow_qoi: xr.Dataset, ambient_ds: xr.Dataset):
    # need to add theta, U, and T to flow_qoi dataset
    # think it has datetime
    ad = ambient_ds
    ds = xr.concat(
        [flow_qoi[qoi], ad[dn.T], ad[dn.wind_speed], ad[dn.wind_dir]], dim=dn.datetime
    )
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3)
    sns.scatterplot(data=ds.to_dataframe(), x=ds[dn.wind_dir], y=ds[qoi], ax=ax1)
    sns.scatterplot(data=ds.to_dataframe(), x=ds[dn.wind_speed], y=ds[qoi], ax=ax2)
    sns.scatterplot(data=ds.to_dataframe(), x=ds[dn.T], y=ds[qoi], ax=ax3)

    # can embed ptpython in here to check things out!

    return fig


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
    g = sns.PairGrid(
        wind_bin.to_dataframe().melt(), corner=True, hue=dn.label
    )  # can think about if want separate correlations per wind direction..
    g.map_lower(sns.scatterplot)
    g.map_lower(annotate_corr)
    g.map_diag(sns.histplot)
    return g.figure
