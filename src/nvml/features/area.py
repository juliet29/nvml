from plyze.flow_graph.interfaces import FlowGraph
from plyze.metrics.calculators import FlowMetricsCalculator, PlanMetricsCalculator
from plyze.metrics.interfaces import MetricHolder
from plyze.qoi.xarray_helpers import CardinalNameLiteral


def calc_plan_area_metrics(G: FlowGraph):
    holder = MetricHolder()
    PlanMetricsCalculator(G, holder)()
    return holder


def calc_flow_data_metrics(G: FlowGraph, drn: CardinalNameLiteral):
    # return create_flow_paths(G)

    holder = MetricHolder()
    FlowMetricsCalculator(G, holder, "NORTH")()
    return holder
