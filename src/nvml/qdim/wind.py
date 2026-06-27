from pathlib import Path

import pandas as pd
import polars as pl
import seaborn.objects as so
import xarray as xr
from loguru import logger
from plyze.flow_graph.interfaces import FlowGraph

from nvml.constants import DataNames as dn
from nvml.qdim.intext import make_int_ext_series
from nvml.utils import dataset_to_polars, save_seaborn_fig


class WindDirectionBins:
    edges = [0, 45, 90, 135, 180, 225, 270, 315, 360]
    labels = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]


def add_wind_sector_coord(ds: xr.Dataset):
    wdb = WindDirectionBins()

    shifted = (ds[dn.wind_dir] + 22.5) % 360  # ← this is what unifies North
    sector = pd.cut(
        shifted, bins=wdb.edges, labels=wdb.labels, include_lowest=True
    ).astype(str)  # pyright: ignore[reportAttributeAccessIssue]
    return ds.assign_coords(
        {dn.wind_sector: (dn.datetime, sector)}
    )  # sector coord indexed along the datetime coord


def prep_comparison_data(
    G: FlowGraph,
    ambient_ds: xr.Dataset,
    qoi_ds: xr.Dataset,
    # ambient_var: str,
    # savedir: Path,
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
    int_ext_df = make_int_ext_series(G)
    df = dataset_to_polars(joined).join(other=int_ext_df, on=dn.space_name)

    return df


def plot(df: pl.DataFrame, savedir: Path, ambient_var: str):
    p = (
        so.Plot(df, x=ambient_var, y=dn.zone_inflow, color=dn.is_external)
        .facet(dn.wind_sector)
        .add(so.Dots())
    )
    # p = (
    #     so.Plot(long, x=ambient_var, y="flow"
    #     )
    #     .add(so.Dots())
    #     .add(so.Line())
    #     .facet(col=dn.wind_sector, row="qoi")
    #     .share(y=False)
    # )
    #
    # n_cols = long[dn.wind_sector].nunique()
    # fig = plt.figure(figsize=(3 * n_cols, 6))
    # p.on(fig).plot()

    path = savedir / f"corr_plot_{ambient_var}.png"
    save_seaborn_fig(p, path)
