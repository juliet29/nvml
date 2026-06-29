from pathlib import Path

import pandas as pd
import polars as pl
import seaborn.objects as so
import xarray as xr
from loguru import logger
from plan2eplus.ezcase.ez import EZ
from plyze.flow_graph.interfaces import FlowGraph
from utils4plans.lists import get_unique_one

from nvml.constants import DataNames as dn
from nvml.qdim.incident import (
    calculate_incidence_factor,
    make_zone_outward_normal_da,
    wind_angle_da_to_vectors,
)
from nvml.qdim.intext import get_normals_for_windows_across_zones, make_int_ext_series
from nvml.utils import save_seaborn_fig, xr_to_polars


class WindDirectionBins:
    edges = [0, 45, 90, 135, 180, 225, 270, 315, 360]
    labels = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]


def add_wind_sector_coord(ds: xr.Dataset):
    wdb = WindDirectionBins()

    shifted = (ds[dn.wind_dir] + 22.5) % 360  # ← this is what unifies North
    sector = pd.cut(
        shifted, bins=wdb.edges, labels=wdb.labels, include_lowest=True
    )  # .astype(str)  # pyright: ignore[reportAttributeAccessIssue]
    return ds.assign_coords(
        {dn.wind_sector: (dn.datetime, sector)}
    )  # sector coord indexed along the datetime coord


# TODO: ensure downsstream qdim works..


def calculate_incidence_factor_for_comparison(
    G: FlowGraph, case: EZ, ambient_ds: xr.Dataset
):
    zons = get_normals_for_windows_across_zones(G, case)
    zone_da = make_zone_outward_normal_da(zons)
    wind_da = wind_angle_da_to_vectors(ambient_ds[dn.wind_dir])
    incidence_factor = calculate_incidence_factor(zone_da, wind_da)
    return incidence_factor


def prep_comparison_data(
    G: FlowGraph,
    ambient_ds: xr.Dataset,
    qoi_ds: xr.Dataset,
    # ambient_var: str,
    # savedir: Path,
):
    """
    ambient_ds should have wind sector information
    qoi_ds is a combination of different variables for one zone
    """
    qois = qoi_ds[[dn.zone_inflow, dn.zone_dimless_flow]]
    ambient = ambient_ds

    # join both data
    joined = xr.merge([qois, ambient], join="inner")
    logger.debug(joined)
    int_ext_df = make_int_ext_series(G)
    df = xr_to_polars(joined).join(other=int_ext_df, on=dn.space_name)

    return df


def add_incidence_data(
    G: FlowGraph, case: EZ, ambient_ds: xr.Dataset, qdim_df: pl.DataFrame
):
    def get_idf_name_by_space_name(space_name: str):
        zone = get_unique_one(G.zone_nodes, lambda x: x.name == space_name)
        return zone.data.idf_name

    if_da = calculate_incidence_factor_for_comparison(G, case, ambient_ds)

    df = xr_to_polars(if_da).with_columns(
        pl.col(dn.space_name).map_elements(
            get_idf_name_by_space_name, return_dtype=pl.String
        )
    )
    logger.debug(df[dn.datetime].dtype)
    logger.debug(qdim_df[dn.datetime].dtype)

    # TODO: get the actual names on the df ..

    full_df = qdim_df.join(other=df, on=[dn.space_name, dn.datetime])
    return full_df


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
