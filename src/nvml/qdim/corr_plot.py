from pathlib import Path

import matplotlib.pyplot as plt
import seaborn.objects as so
import xarray as xr
from loguru import logger

from nvml.constants import DataNames as dn
from nvml.utils import save_mpl_fig


def plot(ds: xr.Dataset, savedir: Path, ambient_var: str):
    df = ds.to_dataframe().reset_index()
    qois = [dn.zone_dimless_flow, dn.zone_inflow]
    long = df.melt(
        id_vars=[c for c in df.columns if c not in qois],
        value_vars=qois,
        var_name="qoi",
        value_name="flow",
    )
    p = (
        so.Plot(long, x=ambient_var, y="flow")
        .add(so.Dots())
        .add(so.Line(), so.PolyFit())
        .facet(col=dn.wind_sector, row="qoi")
        .share(y=False)
    )

    n_cols = long[dn.wind_sector].nunique()
    fig = plt.figure(figsize=(3 * n_cols, 6))
    p.on(fig).plot()

    path = savedir / f"corr_plot_{ambient_var}.png"
    save_mpl_fig(fig, path)


def make_comparison_plot(
    ambient_ds: xr.Dataset,
    qoi_ds: xr.Dataset,
    ambient_var: str,
    savedir: Path,
):
    """
    ambient_ds should have wind sector information
    qoi_ds is a combination of differnt plans
    ambient_var is either temperature or velocity
    """
    # need qdim, zone_inflow from qoi_ds
    qois = qoi_ds[[dn.zone_inflow, dn.zone_dimless_flow]]

    # filter ambient_ds to where matches qoi_ds
    # try:
    #     ambient = ambient_ds.sel({dn.wind_sector: wind_sector})
    # except KeyError as e:
    #     raise ValueError(f"No data for wind sector {wind_sector}: {e}")
    ambient = ambient_ds

    # join both data
    joined = xr.merge([qois, ambient], join="inner")
    logger.debug(joined)

    plot(joined, savedir, ambient_var)
    # hvplot for qdim vs ambinet_var, zone_inflow vs ambient_var
