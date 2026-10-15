from collections import Counter, defaultdict
from math import sqrt

import networkx as nx
import numpy as np
from sklearn.cluster import AgglomerativeClustering

from nvml.explain.interfaces import (
    EMPTY_MOTIF,
    InstanceExplanation,
    RawExplanation,
    Relation,
)
from nvml.gmodel.hetero import CARDINAL_DIRECTIONS, EXTERNAL, FLOW, ZONE

ZONE_FLOW_EXTERNAL = (ZONE, FLOW, EXTERNAL)

DEFAULT_EDGE_QUANTILE = 0.6
DEFAULT_WL_ITERATIONS = 3
DEFAULT_MOTIF_DISTANCE = 0.5
DEFAULT_MAX_EDGES = 4

DEFAULT_AREA_CUTS = (0.2,)

AREA_BIN_LABELS = {2: ["other", "liv"], 3: ["svc", "std", "liv"]}


def zone_bin_letter(
    bin_index: int, n_bins: int, area_labels: tuple[str, ...] | None = None
) -> str:
    if area_labels and len(area_labels) == n_bins:
        return area_labels[bin_index]
    labels = AREA_BIN_LABELS.get(n_bins)
    return labels[bin_index] if labels else str(bin_index)


def normalized_zone_areas(raw: RawExplanation) -> list[float] | None:
    total_area = sum(raw.zone_areas)
    if total_area <= 0:
        return None
    return [area / total_area for area in raw.zone_areas]


def build_important_motif(
    raw: RawExplanation,
    keep_quantile: float,
    area_cuts: tuple[float, ...] | None = None,
    area_labels: tuple[str, ...] | None = None,
    max_edges: int = DEFAULT_MAX_EDGES,
) -> nx.Graph:
    graph = nx.Graph()
    all_importance = [
        value for values in raw.relation_edge_importance.values() for value in values
    ]
    if not all_importance:
        return graph
    threshold = float(np.quantile(np.array(all_importance), keep_quantile))

    normalized_areas = normalized_zone_areas(raw)
    use_area_bins = bool(area_cuts) and normalized_areas is not None
    n_bins = len(area_cuts) + 1 if area_cuts else 0
    graph.graph["area_bins"] = n_bins if use_area_bins else None

    def zone_bin_of(local_index: int) -> int:
        return int(np.digitize(normalized_areas[local_index], area_cuts))

    def zone_label_of(local_index: int) -> str:
        if not use_area_bins:
            return ZONE
        return (
            f"{ZONE}:{zone_bin_letter(zone_bin_of(local_index), n_bins, area_labels)}"
        )

    def label_of(node_type: str, local_index: int) -> str:
        if node_type == EXTERNAL:
            return f"{EXTERNAL}:{raw.external_directions[local_index]}"
        return zone_label_of(local_index)

    def add_typed_node(node_type: str, local_index: int) -> tuple[str, int]:
        key = (node_type, local_index)
        if key not in graph:
            is_zone = node_type == ZONE
            area = raw.zone_areas[local_index] if is_zone else None
            area_norm = (
                normalized_areas[local_index]
                if is_zone and normalized_areas is not None
                else None
            )
            area_bin = zone_bin_of(local_index) if is_zone and use_area_bins else None
            graph.add_node(
                key,
                label=label_of(node_type, local_index),
                area=area,
                area_norm=area_norm,
                area_bin=area_bin,
            )
        return key

    candidates: dict[frozenset, tuple[float, tuple, tuple]] = {}

    def consider(relation: Relation, edges, importances):
        src, _, dst = relation
        for position, importance in enumerate(importances):
            source_local, target_local = edges[position]
            endpoints = ((src, source_local), (dst, target_local))
            key = frozenset(endpoints)
            if key not in candidates or importance > candidates[key][0]:
                candidates[key] = (importance, endpoints[0], endpoints[1])

    for relation, importances in raw.relation_edge_importance.items():
        consider(relation, raw.relation_edges[relation], importances)

    above_threshold = [c for c in candidates.values() if c[0] >= threshold]
    kept = sorted(above_threshold, key=lambda c: c[0], reverse=True)[:max_edges]

    def add_edge(source_endpoint, target_endpoint):
        source = add_typed_node(*source_endpoint)
        target = add_typed_node(*target_endpoint)
        graph.add_edge(source, target)

    for _, source_endpoint, target_endpoint in kept:
        add_edge(source_endpoint, target_endpoint)

    return graph


def extract_instance(
    raw: RawExplanation,
    edge_quantile: float = DEFAULT_EDGE_QUANTILE,
    area_cuts: tuple[float, ...] | None = None,
    wl_iterations: int = DEFAULT_WL_ITERATIONS,
    area_labels: tuple[str, ...] | None = None,
    max_edges: int = DEFAULT_MAX_EDGES,
) -> InstanceExplanation:
    zone_max = max(raw.zone_importance, default=0.0)
    external_max = max(raw.external_importance, default=0.0)
    node_scale = max(zone_max, external_max, 1e-12)

    direction_importance: dict[str, float] = {}

    def accumulate_direction(local_index: int):
        name = raw.external_directions[local_index]
        importance = raw.external_importance[local_index] / node_scale
        direction_importance[name] = direction_importance.get(name, 0.0) + importance

    for local_index in range(len(raw.external_importance)):
        accumulate_direction(local_index)

    zone_area_importance = [
        (area, importance / node_scale)
        for area, importance in zip(raw.zone_areas, raw.zone_importance)
    ]

    zones_by_direction: dict[str, set] = defaultdict(set)
    for zone_index, external_index in raw.relation_edges.get(ZONE_FLOW_EXTERNAL, []):
        zones_by_direction[raw.external_directions[external_index]].add(zone_index)
    cardinal_pathways = {
        direction: len(zones_by_direction.get(direction, ()))
        for direction in CARDINAL_DIRECTIONS
    }

    relation_maxes = [
        max(values) for values in raw.relation_edge_importance.values() if values
    ]
    edge_scale = max(max(relation_maxes, default=0.0), 1e-12)
    relation_importance = {
        relation: (sum(values) / len(values) / edge_scale if values else 0.0)
        for relation, values in raw.relation_edge_importance.items()
    }

    motif_graph = build_important_motif(
        raw, edge_quantile, area_cuts, area_labels, max_edges
    )
    motif_hash = (
        nx.weisfeiler_lehman_graph_hash(
            motif_graph, node_attr="label", iterations=wl_iterations
        )
        if motif_graph.number_of_nodes()
        else EMPTY_MOTIF
    )

    return InstanceExplanation(
        case_name=raw.case_name,
        predicted_class=raw.predicted_class,
        direction_importance=direction_importance,
        relation_importance=relation_importance,
        zone_area_importance=zone_area_importance,
        cardinal_pathways=cardinal_pathways,
        motif_hash=motif_hash,
        motif_graph=motif_graph,
    )


def motif_feature(graph: nx.Graph, wl_iterations: int) -> Counter:
    if graph.number_of_nodes() == 0:
        return Counter()
    subtree_hashes = nx.weisfeiler_lehman_subgraph_hashes(
        graph, node_attr="label", iterations=wl_iterations, include_initial_labels=True
    )
    bag: Counter = Counter()
    for hashes in subtree_hashes.values():
        bag.update(hashes)
    return bag


def motif_cosine_distance(a: Counter, b: Counter) -> float:
    shared = set(a) & set(b)
    dot = sum(a[key] * b[key] for key in shared)
    if dot == 0:
        return 1.0
    norm_a = sqrt(sum(value * value for value in a.values()))
    norm_b = sqrt(sum(value * value for value in b.values()))
    return 1.0 - dot / (norm_a * norm_b)


def cluster_motifs(
    motif_graphs: list[nx.Graph], wl_iterations: int, distance_threshold: float
) -> tuple[list[int], dict[int, int]]:
    features = [motif_feature(graph, wl_iterations) for graph in motif_graphs]
    n = len(features)
    if n == 0:
        return [], {}
    if n == 1:
        return [0], {0: 0}

    distances = np.zeros((n, n))

    def fill(i: int, j: int):
        distance = motif_cosine_distance(features[i], features[j])
        distances[i, j] = distance
        distances[j, i] = distance

    for i in range(n):
        for j in range(i + 1, n):
            fill(i, j)

    model = AgglomerativeClustering(
        metric="precomputed",
        linkage="average",
        distance_threshold=distance_threshold,
        n_clusters=None,
    )
    labels = [int(label) for label in model.fit_predict(distances)]

    members_of: dict[int, list[int]] = defaultdict(list)
    for index, label in enumerate(labels):
        members_of[label].append(index)

    representative: dict[int, int] = {}

    def pick_medoid(members: list[int]) -> int:
        return min(members, key=lambda i: sum(distances[i, j] for j in members))

    for label, members in members_of.items():
        representative[label] = pick_medoid(members)

    return labels, representative
