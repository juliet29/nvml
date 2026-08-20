import torch
import torch_geometric.transforms as T
from plyze.flow_graph.interfaces import FlowGraph
from torch_geometric.data import HeteroData

from nvml.constants import DataNames

CARDINAL_DIRECTIONS = ["NORTH", "SOUTH", "EAST", "WEST"]

ZONE = "zone"
EXTERNAL = "external"
FLOW = "flow"

ZONE_FEATURE_DIM = 1
EXTERNAL_FEATURE_DIM = len(CARDINAL_DIRECTIONS)

NODE_TYPES = [ZONE, EXTERNAL]
RELATIONS = [
    (ZONE, FLOW, ZONE),
    (ZONE, FLOW, EXTERNAL),
    (EXTERNAL, f"rev_{FLOW}", ZONE),
]

FEATURE_DIM_OF = {ZONE: ZONE_FEATURE_DIM, EXTERNAL: EXTERNAL_FEATURE_DIM}


def flow_graph_to_hetero_data(graph: FlowGraph, case_name: str) -> HeteroData:
    zone_position = {node.name: ix for ix, node in enumerate(graph.zone_nodes)}
    external_position = {node.name: ix for ix, node in enumerate(graph.external_nodes)}
    direction_position = {name: ix for ix, name in enumerate(CARDINAL_DIRECTIONS)}

    def encode_direction(name: str):
        if name not in direction_position:
            raise ValueError(
                f"External node {name!r} in {case_name} is not a known cardinal direction"
            )
        row = [0.0] * EXTERNAL_FEATURE_DIM
        row[direction_position[name]] = 1.0
        return row

    zone_to_zone: list[tuple[int, int]] = []
    zone_to_external: list[tuple[int, int]] = []

    def route(u: str, v: str):
        if u in zone_position and v in zone_position:
            zone_to_zone.append((zone_position[u], zone_position[v]))
        elif u in zone_position and v in external_position:
            zone_to_external.append((zone_position[u], external_position[v]))
        elif v in zone_position and u in external_position:
            zone_to_external.append((zone_position[v], external_position[u]))
        else:
            raise ValueError(f"Edge {u}-{v} in {case_name} connects two external nodes")

    for edge in graph.edges_with_data:
        route(edge.u, edge.v)

    def to_edge_index(pairs: list[tuple[int, int]]):
        if not pairs:
            return torch.empty((2, 0), dtype=torch.long)
        return torch.tensor(pairs, dtype=torch.long).t().contiguous()

    data = HeteroData()
    data[ZONE].x = torch.tensor(
        [[node.data.area] for node in graph.zone_nodes], dtype=torch.float
    )
    data[EXTERNAL].x = torch.tensor(
        [encode_direction(name) for name in graph.external_node_names],
        dtype=torch.float,
    )
    data[ZONE, FLOW, ZONE].edge_index = to_edge_index(zone_to_zone)
    data[ZONE, FLOW, EXTERNAL].edge_index = to_edge_index(zone_to_external)

    setattr(data, DataNames.case_name, case_name)
    return T.ToUndirected()(data)
