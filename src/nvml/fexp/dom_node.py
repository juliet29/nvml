from typing import get_args

import networkx as nx
from plyze.flow_graph.interfaces import FlowGraph
from plyze.metrics.helpers.flow_paths import organize_external_nodes
from plyze.qoi.xarray_helpers import CardinalNameLiteral


def distinguished_external_nodes(G: FlowGraph, drn: CardinalNameLiteral):
    try:
        nodes = organize_external_nodes(G.external_nodes, drn)

    except ValueError:
        # logger.critical(f"Error organizing nodes: {e}")
        return None
    return nodes


def calc_degree_dominant_node(G: FlowGraph, drn: CardinalNameLiteral):
    dnodes = distinguished_external_nodes(G, drn)
    if not dnodes:
        return 0
    deg = nx.degree(G, dnodes.main.name)
    return deg


def calc_degree_dominant_node_all_drns(G: FlowGraph):
    res = {
        drn: calc_degree_dominant_node(G, drn) for drn in get_args(CardinalNameLiteral)
    }
    return res
