import networkx as nx
import xarray as xr

## flow paths => for now, assuming dominant external node is well-chosen by segmenting on wind direction. #TODO: investigate!
# get flow along the path at a specific time..  => need to show that its time invariant for a specific wind direction..
# that means need qdim for an edge..
from plyze.flow_graph.interfaces import Edge, FlowGraph
from utils4plans.lists import chain_flatten

from nvml.constants import DataNames as dn


class PathGraph(nx.DiGraph):
    @classmethod
    def create(cls, path: list[str]):
        G = nx.path_graph(path, create_using=nx.DiGraph)
        return cls(G)

    @property
    def edge_list(self):
        return list(self.edges)

    # def frozen_edge_list(self):
    #     return list([frozenset(i) for i in self.edges])


def get_unique_edges(path_graphs: list[PathGraph]):
    all_edges = chain_flatten([i.edge_list for i in path_graphs])
    return [frozenset(i) for i in set(all_edges)]


def prep_edge_qdim(edge: Edge, velocity: xr.DataArray):
    """
    Velocity datetimes should be filtered to one wind sector before being passed in
    """
    q = edge.data.flow_in.sel({dn.datetime: velocity[dn.datetime]})
    q_dim = q / (velocity * edge.data.surface_area)
    return q_dim


def get_relevant_edges(G: FlowGraph, path_graphs: list[PathGraph]):
    # NOTE: may include all edges, so might actually not neede this..
    # ic(pretty_repr(G.sorted_edge_names))
    unique_edges = get_unique_edges(path_graphs)
    # ic(pretty_repr(sorted(unique_edges)))
    # ic((len(G.edges_with_data), len(unique_edges)))
    res = [i for i in G.edges_with_data if frozenset(i.as_tuple) in unique_edges]
    # ic(pretty_repr(sorted([i.as_tuple for i in res])))

    assert len(res) == len(unique_edges), (
        f"Expcted {len(unique_edges)} edges but instead got {len(res)}"
    )
    return res
