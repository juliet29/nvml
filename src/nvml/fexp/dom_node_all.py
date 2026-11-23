from typing import NamedTuple, get_args

from plyze.flow_graph.interfaces import FlowGraph
from plyze.metrics.calculators import FlowMetricsCalculator
from plyze.metrics.interfaces import MetricHolder
from plyze.metrics.registries import FlowMetricRegistry as fmr
from plyze.qoi.xarray_helpers import CardinalNameLiteral


class FlowPaths(NamedTuple):
    num_paths: int
    avg_path_length: float
    len_shortest_path: int
    degree_dom_node: int


def calc_flow_data_metrics_per_drn(G: FlowGraph, drn: CardinalNameLiteral):
    holder = MetricHolder()
    FlowMetricsCalculator(G, holder, drn)()  # pyright: ignore[reportArgumentType]
    d = holder.holder_dict

    return FlowPaths(
        d[fmr.num_paths.name],  # pyright: ignore[reportArgumentType]
        d[fmr.avg_path_length.name],
        d[fmr.len_shortest_path.name],  # pyright: ignore[reportArgumentType]
        d[fmr.degree_dom_node.name],  # pyright: ignore[reportArgumentType]
    )


def calc_dom_node_flow_metrics_for_all_drns(G: FlowGraph):
    res = {
        drn: calc_flow_data_metrics_per_drn(G, drn)
        for drn in get_args(CardinalNameLiteral)
    }
    return res
