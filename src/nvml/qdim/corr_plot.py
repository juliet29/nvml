import matplotlib.pyplot as plt
import xarray as xr
from loguru import logger

from nvml.constants import DataNames as dn


def make_comparison_plot(ds: xr.DataArray):
    fig, axes = plt.subplots(1, 2)
    ds.plot.scatter(x=dn.t_out, y=dn.zone_dimless_flow, ax=axes[0])
    ds.plot.scatter(x=dn.t_out, y=dn.zone_inflow, ax=axes[1])
    # TODO: add to utils4plans -> save mpl fig, make_dir


def plot_comparison_corr_plot(
    ambient_ds: xr.Dataset, qoi_ds: xr.Dataset, wind_sector: str, ambient_var: str
):
    """
    ambient_ds should have wind sector information
    qoi_ds is a combination of differnt plans
    ambient_var is either temperature or velocity
    """
    # need qdim, zone_inflow from qoi_ds
    qois = qoi_ds[[dn.zone_inflow, dn.zone_dimless_flow]]

    # filter ambient_ds to where matches qoi_ds
    try:
        ambient = ambient_ds.sel({dn.wind_sector: wind_sector})
    except KeyError as e:
        raise ValueError(f"No data for wind sector {wind_sector}: {e}")

    # join both data
    joined = xr.merge([qois, ambient], join="inner")
    logger.debug(joined)

    # hvplot for qdim vs ambinet_var, zone_inflow vs ambient_var
