import polars as pl
from plan2eplus.ezcase.ez import EZ
from plan2eplus.geometry.directions import WallNormal
from plan2eplus.ops.subsurfaces.ezobject import Subsurface
from plyze.flow_graph.interfaces import Edge, FlowGraph, ZoneNode
from rich.pretty import pretty_repr
from utils4plans.lists import get_unique_one

from nvml.constants import DataNames
from nvml.qdim.interfaces import EdgeAndNormal, ZoneAndOutwardNormals

"""Properties derived from the graph, idf, or both"""

CardinalNames = ["EAST", "SOUTH", "NORTH", "WEST"]
# TODO: redundant, could use WallNormal values, .. but explicit


def is_external(G: FlowGraph, node: str):
    nbs = list(G.neighbors(node))
    for n in nbs:
        if n in CardinalNames:
            return True
    return False


def make_int_ext_series(G: FlowGraph):
    def handle(node: ZoneNode):
        return {
            DataNames.space_name: node.data.idf_name,
            DataNames.is_external: is_external(G, node.name),
        }

    ds = [handle(i) for i in G.zone_nodes]

    return pl.DataFrame(ds)


# TODO: move -> calculating zone normal values


def find_subsurface_by_edge(candidate_surfaces: list[Subsurface], edge: Edge):
    cs_edges = [s.edge for s in candidate_surfaces]
    for s in candidate_surfaces:
        sa, sb = s.edge
        u, v = edge.u, edge.v
        if sa == u and sb == v:
            return s
    raise LookupError(f"Could not find a match for {edge} in {pretty_repr(cs_edges)}")


def get_zone_outward_normals(
    case: EZ, G: FlowGraph, node: ZoneNode, potential_windows: list[Subsurface]
):
    def get(edge: Edge):
        s = find_subsurface_by_edge(candidate_subsurfaces, edge)
        wall_normal = s.surface.direction
        assert isinstance(wall_normal, WallNormal), (
            f"Expected surface {s.surface} attatched to {edge.u, edge.v} to have type of WallNormal, instead got type {type(wall_normal)}"
        )
        return wall_normal

    edge_names = list(G.edges(node.name))
    edges = [i for i in G.edges_with_data if i in edge_names]
    window_edges = [i for i in edges if i.data.surface_type == "Window"]

    zone = get_unique_one(
        case.objects.zones, lambda x: x.zone_name.upper() == node.data.idf_name
    )
    candidate_subsurfaces = [
        i for i in potential_windows if i.subsurface_name in zone.subsurface_names
    ]
    res = [EdgeAndNormal(e.u, e.v, get(e)) for e in window_edges]
    return res


def get_normals_for_windows_across_zones(G: FlowGraph, case: EZ):
    windows = [i for i in case.objects.subsurfaces if i.subsurface_type == "Window"]

    external_zones = [i for i in G.zone_nodes if is_external(G, i.name)]
    zone_outward_normals = [
        ZoneAndOutwardNormals(i.name, get_zone_outward_normals(case, G, i, windows))
        for i in external_zones
    ]
    return zone_outward_normals
