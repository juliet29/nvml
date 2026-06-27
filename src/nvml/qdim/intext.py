from pathlib import Path

import polars as pl
from plan2eplus.ezcase.ez import EZ
from plyze.flow_graph.interfaces import FlowGraph, ZoneNode

from nvml.constants import DataNames

"""Properties derived from the graph, idf, or both"""

CardinalNames = ["EAST", "SOUTH", "NORTH", "WEST"]


def is_external(G: FlowGraph, node: str):
    nbs = list(G.neighbors(node))
    for n in nbs:
        if n in CardinalNames:
            return True
    return False


def make_int_ext_series(G: FlowGraph):
    def handle(node: ZoneNode):
        return {
            DataNames.space_name: node.data.idf_name,
            DataNames.is_external: is_external(G, node.name),
        }

    ds = [handle(i) for i in G.zone_nodes]

    return pl.DataFrame(ds)


# incident angles
def get_subsurface_normals(G: FlowGraph, idf_path: Path):
    case = EZ(idf_path)

    pass
