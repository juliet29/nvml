from pathlib import Path

import xarray as xr
from plyze import FlowGraphModel
from plyze.flow_graph.create.main import make_ambient_data
from plyze.flow_graph.interfaces import FlowGraph
from plyze.qoi_flow_graph.zone_data import collate_ambient_data, collate_zone_data

# PathOrData = Data | str | Path


def read_graph(x: Path | FlowGraph):
    if isinstance(x, Path):
        return FlowGraphModel.read(x)
    return x


def graph_to_ds(x: Path | FlowGraph):
    def handle(x):
        G = read_graph(x)
        ds = collate_zone_data(G)
        return ds

    return handle(x)


def get_ambient_data_as_ds(x: Path | xr.Dataset):
    """path is eplusout.sql"""
    if isinstance(x, Path):
        res = make_ambient_data(x)
        return collate_ambient_data(res)
    elif isinstance(x, xr.Dataset):
        # TODO: check that it has the correct properties..
        return x
