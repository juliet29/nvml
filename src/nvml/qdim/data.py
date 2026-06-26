from pathlib import Path

import pandas as pd
import xarray as xr
from plyze import FlowGraphModel
from plyze.flow_graph.create.main import make_ambient_data
from plyze.qoi_flow_graph.zone_data import collate_ambient_data, collate_zone_data

from nvml.constants import DataNames as dn


# TODO: move io stuff elsewhere
def graph_to_ds(path: Path):
    def handle(path: Path):
        G = FlowGraphModel.read(path)
        ds = collate_zone_data(G)
        return ds

    return handle(path)


def get_ambient_data_as_ds(path: Path):
    """path is eplusout.sql"""
    res = make_ambient_data(path)
    return collate_ambient_data(res)


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
