from typing import NamedTuple

import networkx as nx
import pandas as pd
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

    def get_qdim_data(self, da: xr.DataArray):
        # this will be organized as it is called
        return da.sel({dn.edge_name: self.edge_list})

    # def frozen_edge_list(self):
    #     return list([frozenset(i) for i in self.edges])


def get_unique_edges(path_graphs: list[PathGraph]):
    all_edges = chain_flatten([i.edge_list for i in path_graphs])
    # if frozen:
    #     return [frozenset(i) for i in set(all_edges)]
    return list(set(all_edges))


def prep_edge_qdim(edge: Edge, velocity: xr.DataArray, is_flipped: bool):
    """
    Velocity datetimes should be filtered to one wind sector before being passed in
    """
    ed = edge.data

    q = ed.flow_in if is_flipped else ed.flow_in
    qt = q.sel({dn.datetime: velocity[dn.datetime]})
    q_dim = qt / (velocity * edge.data.surface_area)
    return q_dim.drop_vars(dn.space_name)


def check_all_path_edges_found(q_dims: list, unique_edges: list):
    assert len(q_dims) == len(unique_edges), (
        f"Expcted {len(unique_edges)} edges but instead got {len(q_dims)}"
    )


def prep_edge_da(G: FlowGraph, path_graphs: list[PathGraph], wind_speed: xr.DataArray):
    class EdgeInfo(NamedTuple):
        edge_ix_name: tuple[str, str]
        q_dim: xr.DataArray

    def handle(edge: Edge):
        if edge.as_tuple in unique_edges:
            is_flipped = False
            e_ix = edge.as_tuple
        elif edge.as_tuple_reverse in unique_edges:
            is_flipped = True
            e_ix = edge.as_tuple_reverse
        q_dim = prep_edge_qdim(edge, wind_speed, is_flipped)
        return EdgeInfo(e_ix, q_dim)

    unique_edges = get_unique_edges(path_graphs)
    # ic(sorted(unique_edges))
    # ic(sorted([i.as_tuple for i in G.edges_with_data]))
    # breakpoint()
    einfo = [handle(e) for e in G.edges_with_data]
    check_all_path_edges_found(einfo, unique_edges)

    # index
    midx = pd.MultiIndex.from_tuples(
        [e.edge_ix_name for e in einfo], names=(dn.u, dn.v)
    )
    midx_coords = xr.Coordinates.from_pandas_multiindex(midx, dim=dn.edge_name)
    qdims = [i.q_dim for i in einfo]

    da = xr.concat(qdims, dim=dn.edge_name).assign_coords(midx_coords)
    return da
