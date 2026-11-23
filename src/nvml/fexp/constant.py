from typing import NamedTuple

from plyze.flow_graph.interfaces import FlowGraph
from plyze.metrics.calculators import PlanMetricsCalculator
from plyze.metrics.interfaces import MetricHolder
from plyze.metrics.registries import PlanMetricRegistry as pmr


class BulkPlan(NamedTuple):
    area: float
    num_rooms: int


# class FlowPaths(NamedTuple):
#     num_paths: int
#     avg_path_length: float
#     len_shortest_path: int
#


def calc_plan_area_metrics(G: FlowGraph):
    holder = MetricHolder()
    PlanMetricsCalculator(G, holder)()
    d = holder.holder_dict
    return BulkPlan(d[pmr.area.name], d[pmr.num_rooms.name])  # pyright: ignore[reportArgumentType]


# def calc_flow_data_metrics(G: FlowGraph):
#     holder = MetricHolder()
#     FlowMetricsCalculator(G, holder, None)()  # pyright: ignore[reportArgumentType]
#     d = holder.holder_dict
#
#     return FlowPaths(
#         d[fmr.num_paths.name],  # pyright: ignore[reportArgumentType]
#         d[fmr.avg_path_length.name],
#         d[fmr.mode_path_length.name],  # pyright: ignore[reportArgumentType]
#         d[fmr.len_longest_path.name],  # pyright: ignore[reportArgumentType]
#         d[fmr.len_shortest_path.name],  # pyright: ignore[reportArgumentType]
#     )
