import xarray as xr
from utils4plots import CustomColorObjects

from nvml.constants import DataNames as dn


def q_theta_plot(da: xr.DataArray):
    n_rooms = 7
    # fig, ax = plt.subplots(1, 2, figsize=(7, 4))
    da_small = da.isel({dn.case_name: range(6)}).isel(
        {dn.space_ix: range(n_rooms)}
    )  # .plot.scatter()
    f = da_small.plot.scatter(
        x=dn.wind_dir,
        col=dn.case_name,
        hue=dn.space_ix,
        levels=n_rooms,
        col_wrap=3,
        # ylim=(0, 1.5),
        s=50,
        # edgecolors="face",
        alpha=0.5,
        cmap=CustomColorObjects.cmap,
    )
    return f.fig
