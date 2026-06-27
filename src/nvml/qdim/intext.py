import polars as pl
from plyze.flow_graph.interfaces import FlowGraph, ZoneNode

from nvml.constants import DataNames

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
