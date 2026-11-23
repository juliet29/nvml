from typing import NamedTuple

import numpy as np
from plyze.flow_graph.interfaces import FlowGraph
from plyze.metrics.calculators import PlanMetricsCalculator
from plyze.metrics.interfaces import MetricHolder
from plyze.metrics.registries import PlanMetricRegistry as pmr


class BulkPlan(NamedTuple):
    area: float
    num_rooms: int


class GraphTheory(NamedTuple):
    avg_edges_per_node: float
    max_degree: float  # hi
    avg_degree: float  # hi


def calc_plan_area_metrics(G: FlowGraph):
    holder = MetricHolder()
    PlanMetricsCalculator(G, holder)()
    d = holder.holder_dict
    return BulkPlan(d[pmr.area.name], d[pmr.num_rooms.name])  # pyright: ignore[reportArgumentType]


# calc graph metrics..
#
def calc_graph_theory_metrics(G: FlowGraph):
    epn = len(G.edges) / len(G.nodes)
    dg = np.array([G.degree(i) for i in G.nodes])
    max_deg = np.max(dg)
    avg_deg = np.max(dg)

    return GraphTheory(epn, max_deg, avg_deg)
