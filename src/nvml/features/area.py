from plan2eplus.geometry.directions import WallNormal
from plyze.flow_graph.interfaces import FlowGraph
from plyze.metrics.calculators import FlowMetricsCalculator, PlanMetricsCalculator
from plyze.metrics.interfaces import MetricHolder
from plyze.qoi.xarray_helpers import CardinalNames


def calc_plan_area_metrics(G: FlowGraph):
    holder = MetricHolder()
    PlanMetricsCalculator(G, holder)()
    return holder.holder_dict


def calc_flow_data_metrics(G: FlowGraph, wall_normal: WallNormal):
    holder = MetricHolder()
    assert wall_normal.name in CardinalNames
    FlowMetricsCalculator(G, holder, wall_normal.name)()  # pyright: ignore[reportArgumentType]
    return holder.holder_dict
