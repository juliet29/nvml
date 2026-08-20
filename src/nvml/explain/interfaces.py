from dataclasses import dataclass

import networkx as nx

Relation = tuple[str, str, str]

EMPTY_MOTIF = "empty"


@dataclass
class RawExplanation:
    case_name: str
    predicted_class: int
    zone_areas: list[float]
    zone_importance: list[float]
    external_directions: list[str]
    external_importance: list[float]
    relation_edges: dict[Relation, list[tuple[int, int]]]
    relation_edge_importance: dict[Relation, list[float]]


@dataclass
class InstanceExplanation:
    case_name: str
    predicted_class: int
    direction_importance: dict[str, float]
    relation_importance: dict[Relation, float]
    zone_area_importance: list[tuple[float, float]]
    motif_hash: str
    motif_graph: nx.Graph


@dataclass
class MotifPrevalence:
    motif_hash: str
    representative_graph: nx.Graph
    count: int
    frequency: float
    lift: float


@dataclass
class ClassExplanation:
    predicted_class: int
    n_instances: int
    direction_importance: dict[str, float]
    direction_coverage: dict[str, float]
    relation_importance: dict[Relation, float]
    zone_area_importance: list[tuple[float, float]]
    motifs: list[MotifPrevalence]
