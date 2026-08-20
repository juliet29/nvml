from collections import defaultdict

import networkx as nx

from nvml.explain.interfaces import (
    EMPTY_MOTIF,
    ClassExplanation,
    InstanceExplanation,
    MotifPrevalence,
)
from nvml.gmodel.hetero import CARDINAL_DIRECTIONS


def aggregate_explanations(
    instances: list[InstanceExplanation],
) -> dict[int, ClassExplanation]:
    by_class: dict[int, list[InstanceExplanation]] = defaultdict(list)
    for instance in instances:
        by_class[instance.predicted_class].append(instance)

    total_instances = len(instances)
    motif_total_counts: dict[str, int] = defaultdict(int)
    motif_representative: dict[str, nx.Graph] = {}

    def register_motif(instance: InstanceExplanation):
        if instance.motif_hash == EMPTY_MOTIF:
            return
        motif_total_counts[instance.motif_hash] += 1
        motif_representative.setdefault(instance.motif_hash, instance.motif_graph)

    for instance in instances:
        register_motif(instance)

    result: dict[int, ClassExplanation] = {}

    def summarize_class(predicted_class: int, class_instances: list[InstanceExplanation]):
        n = len(class_instances)

        direction_importance: dict[str, float] = {}
        direction_coverage: dict[str, float] = {}

        def summarize_direction(direction: str):
            present = [
                i.direction_importance[direction]
                for i in class_instances
                if direction in i.direction_importance
            ]
            direction_coverage[direction] = len(present) / n
            direction_importance[direction] = sum(present) / len(present) if present else 0.0

        for direction in CARDINAL_DIRECTIONS:
            summarize_direction(direction)

        relation_totals: dict[tuple, float] = defaultdict(float)
        relation_seen: dict[tuple, int] = defaultdict(int)

        def accumulate_relations(instance: InstanceExplanation):
            for relation, value in instance.relation_importance.items():
                relation_totals[relation] += value
                relation_seen[relation] += 1

        for instance in class_instances:
            accumulate_relations(instance)
        relation_importance = {
            relation: relation_totals[relation] / relation_seen[relation]
            for relation in relation_totals
        }

        zone_area_importance = [
            pair for instance in class_instances for pair in instance.zone_area_importance
        ]

        class_motif_counts: dict[str, int] = defaultdict(int)
        for instance in class_instances:
            if instance.motif_hash != EMPTY_MOTIF:
                class_motif_counts[instance.motif_hash] += 1

        motifs: list[MotifPrevalence] = []

        def register_prevalence(motif_hash: str, count: int):
            frequency = count / n
            base_rate = motif_total_counts[motif_hash] / total_instances
            motifs.append(
                MotifPrevalence(
                    motif_hash=motif_hash,
                    representative_graph=motif_representative[motif_hash],
                    count=count,
                    frequency=frequency,
                    lift=frequency / base_rate,
                )
            )

        for motif_hash, count in class_motif_counts.items():
            register_prevalence(motif_hash, count)
        motifs.sort(key=lambda m: (m.lift, m.count), reverse=True)

        result[predicted_class] = ClassExplanation(
            predicted_class=predicted_class,
            n_instances=n,
            direction_importance=direction_importance,
            direction_coverage=direction_coverage,
            relation_importance=relation_importance,
            zone_area_importance=zone_area_importance,
            motifs=motifs,
        )

    for predicted_class, class_instances in sorted(by_class.items()):
        summarize_class(predicted_class, class_instances)

    return result
