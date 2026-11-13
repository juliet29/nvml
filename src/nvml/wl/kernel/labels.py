from typing import Literal, get_args

import networkx as nx
from plyze.flow_graph.interfaces import FlowGraph


class LabelTags:
    NODE_LABEL = "node_label"
    EDGE_LABEL = "edge_label"


def null_node_labels(G: FlowGraph):
    d = {i: 1 for i in list(G.nodes)}
    nx.set_node_attributes(G, d, LabelTags.NODE_LABEL)
    return G


def external_internal_node_label(G: FlowGraph):
    d = {i: 1 for i in list(G.nodes) if i in G.zone_names}
    d.update({i: 2 for i in list(G.nodes) if i in G.external_node_names})
    nx.set_node_attributes(G, d, LabelTags.NODE_LABEL)
    return G


def area_ratio_map_calc(G: FlowGraph):
    total_area = sum([i.data.area for i in G.zone_nodes])
    area_map = {i.name: i.data.area / total_area for i in G.zone_nodes}
    return area_map


def area_ratio_node_label(G: FlowGraph):
    area_map = area_ratio_map_calc(G)
    area_map.update({i: 0 for i in list(G.nodes) if i in G.external_node_names})
    nx.set_node_attributes(G, area_map, LabelTags.NODE_LABEL)
    return G


def discrete_area_ratio_node_label(G: FlowGraph):
    area_map = area_ratio_map_calc(G)
    sz_sort = sorted(G.zone_names, key=lambda x: area_map[x])
    discrete_map = {name: ix for ix, name in enumerate(sz_sort)}
    discrete_map.update({i: 0 for i in list(G.nodes) if i in G.external_node_names})
    nx.set_node_attributes(G, discrete_map, LabelTags.NODE_LABEL)
    return G


def null_edge_labels(G: FlowGraph):
    d = {i: 1 for i in list(G.edges)}
    nx.set_edge_attributes(G, d, LabelTags.EDGE_LABEL)
    return G


def external_internal_edge_label(G: FlowGraph):
    d = {i.as_tuple: 1 for i in G.edges_with_data if not i.is_external}
    d.update({i.as_tuple: 2 for i in G.edges_with_data if i.is_external})
    nx.set_edge_attributes(G, d, LabelTags.EDGE_LABEL)
    return G


LabelSchemeNames = Literal["null", "int_ext", "area_ratio", "discrete_area_ratio"]


class LabelSchemes:
    def get_method(self, name: LabelSchemeNames):
        return self.__getattribute__(name)

    @classmethod
    def null(cls, G: FlowGraph):
        g1 = null_node_labels(G)
        return null_edge_labels(g1)

    @classmethod
    def int_ext(cls, G: FlowGraph):
        g1 = external_internal_node_label(G)
        return external_internal_edge_label(g1)

    @classmethod
    def area_ratio(cls, G: FlowGraph):
        g1 = area_ratio_node_label(G)
        return external_internal_edge_label(g1)

    @classmethod
    def discrete_area_ratio(cls, G: FlowGraph):
        g1 = discrete_area_ratio_node_label(G)
        return external_internal_edge_label(g1)


def try_all_labels(G: FlowGraph):
    res: dict[LabelSchemeNames, FlowGraph] = {}
    ls = LabelSchemes()
    for name in get_args(LabelSchemeNames):
        scheme_fx = ls.get_method(name)
        res[name] = scheme_fx(G)
    return res


## There is also a world where drop the external nodes completely so can focus on internal structure, but think can control for that with the regression
