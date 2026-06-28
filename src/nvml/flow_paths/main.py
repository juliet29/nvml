import networkx as nx
import xarray as xr

## flow paths => for now, assuming dominant external node is well-chosen by segmenting on wind direction. #TODO: investigate!
# get flow along the path at a specific time..  => need to show that its time invariant for a specific wind direction..
# that means need qdim for an edge..
from plyze.flow_graph.interfaces import FlowGraph
from utils4plans.lists import chain_flatten


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
    return set(all_edges)


def prep_edge_qdim(edge: Edge, velocity: xr.DataArray, dt_coords: xr.Coordinates):
    pass


def get_relevant_edges(G: FlowGraph, path_graphs: list[PathGraph]):
    unique_edges = get_unique_edges(path_graphs)
    res = [i for i in G.edges_with_data if (i.u, i.v) in unique_edges]
    assert len(res) == len(unique_edges)
