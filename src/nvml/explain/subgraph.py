import networkx as nx
import numpy as np

from nvml.explain.interfaces import (
    EMPTY_MOTIF,
    InstanceExplanation,
    RawExplanation,
    Relation,
)
from nvml.gmodel.hetero import EXTERNAL, ZONE

DEFAULT_EDGE_QUANTILE = 0.6


def build_important_motif(raw: RawExplanation, keep_quantile: float) -> nx.Graph:
    graph = nx.Graph()
    all_importance = [
        value for values in raw.relation_edge_importance.values() for value in values
    ]
    if not all_importance:
        return graph
    threshold = float(np.quantile(np.array(all_importance), keep_quantile))

    def label_of(node_type: str, local_index: int) -> str:
        if node_type == EXTERNAL:
            return f"{EXTERNAL}:{raw.external_directions[local_index]}"
        return ZONE

    def add_typed_node(node_type: str, local_index: int) -> tuple[str, int]:
        key = (node_type, local_index)
        if key not in graph:
            area = raw.zone_areas[local_index] if node_type == ZONE else None
            graph.add_node(key, label=label_of(node_type, local_index), area=area)
        return key

    def add_relation(relation: Relation, edges, importances):
        src, _, dst = relation

        def register(edge_position: int):
            source_local, target_local = edges[edge_position]
            source = add_typed_node(src, source_local)
            target = add_typed_node(dst, target_local)
            graph.add_edge(source, target)

        for edge_position, importance in enumerate(importances):
            if importance >= threshold:
                register(edge_position)

    for relation, importances in raw.relation_edge_importance.items():
        add_relation(relation, raw.relation_edges[relation], importances)

    return graph


def extract_instance(
    raw: RawExplanation, edge_quantile: float = DEFAULT_EDGE_QUANTILE
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

    relation_maxes = [
        max(values) for values in raw.relation_edge_importance.values() if values
    ]
    edge_scale = max(max(relation_maxes, default=0.0), 1e-12)
    relation_importance = {
        relation: (sum(values) / len(values) / edge_scale if values else 0.0)
        for relation, values in raw.relation_edge_importance.items()
    }

    motif_graph = build_important_motif(raw, edge_quantile)
    motif_hash = (
        nx.weisfeiler_lehman_graph_hash(motif_graph, node_attr="label")
        if motif_graph.number_of_nodes()
        else EMPTY_MOTIF
    )

    return InstanceExplanation(
        case_name=raw.case_name,
        predicted_class=raw.predicted_class,
        direction_importance=direction_importance,
        relation_importance=relation_importance,
        zone_area_importance=zone_area_importance,
        motif_hash=motif_hash,
        motif_graph=motif_graph,
    )
