from pathlib import Path

import xarray as xr
from plyze.flow_graph.interfaces import FlowGraph

from nvml.io import graph_to_ds


def calc_qdim(
    graph: Path | FlowGraph,
    ambient_ds: xr.Dataset,
):
    pass

    qoi_ds = graph_to_ds(graph)
    return qoi_ds
