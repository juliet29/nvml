from collections import Counter, defaultdict

from nvml.explain.interfaces import (
    ClassExplanation,
    InstanceExplanation,
    MotifPrevalence,
)
from nvml.explain.subgraph import (
    DEFAULT_MOTIF_DISTANCE,
    DEFAULT_WL_ITERATIONS,
    cluster_motifs,
)
from nvml.gmodel.hetero import CARDINAL_DIRECTIONS

DEFAULT_MIN_SUPPORT = 0.05


def aggregate_explanations(
    instances: list[InstanceExplanation],
    motif_distance: float = DEFAULT_MOTIF_DISTANCE,
    wl_iterations: int = DEFAULT_WL_ITERATIONS,
    min_support: float = DEFAULT_MIN_SUPPORT,
) -> dict[int, ClassExplanation]:
    total_instances = len(instances)

    non_empty_indices = [
        index for index, instance in enumerate(instances)
        if instance.motif_graph.number_of_nodes()
    ]
    family_labels, medoid_of = cluster_motifs(
        [instances[index].motif_graph for index in non_empty_indices],
        wl_iterations,
        motif_distance,
    )
    family_of_instance = {
        non_empty_indices[position]: label
        for position, label in enumerate(family_labels)
    }
    family_representative = {
        label: instances[non_empty_indices[medoid_position]].motif_graph
        for label, medoid_position in medoid_of.items()
    }
    family_representative_case = {
        label: instances[non_empty_indices[medoid_position]].case_name
        for label, medoid_position in medoid_of.items()
    }
    family_total_counts = Counter(family_labels)

    by_class: dict[int, list[int]] = defaultdict(list)
    for index, instance in enumerate(instances):
        by_class[instance.predicted_class].append(index)

    result: dict[int, ClassExplanation] = {}

    def summarize_class(predicted_class: int, member_indices: list[int]):
        class_instances = [instances[index] for index in member_indices]
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

        pathway_counts = {
            direction: [instance.cardinal_pathways.get(direction, 0) for instance in class_instances]
            for direction in CARDINAL_DIRECTIONS
        }

        family_counts = Counter(
            family_of_instance[index]
            for index in member_indices
            if index in family_of_instance
        )

        motifs: list[MotifPrevalence] = []

        def register_family(family_label: int, count: int):
            frequency = count / n
            family_total = family_total_counts[family_label]
            motifs.append(
                MotifPrevalence(
                    motif_hash=str(family_label),
                    representative_graph=family_representative[family_label],
                    representative_case=family_representative_case[family_label],
                    count=count,
                    frequency=frequency,
                    lift=frequency / (family_total / total_instances),
                    exclusivity=count / family_total,
                )
            )

        for family_label, count in family_counts.items():
            register_family(family_label, count)
        motifs.sort(
            key=lambda m: (m.frequency >= min_support, m.exclusivity, m.count),
            reverse=True,
        )

        result[predicted_class] = ClassExplanation(
            predicted_class=predicted_class,
            n_instances=n,
            direction_importance=direction_importance,
            direction_coverage=direction_coverage,
            relation_importance=relation_importance,
            zone_area_importance=zone_area_importance,
            pathway_counts=pathway_counts,
            motifs=motifs,
        )

    for predicted_class, member_indices in sorted(by_class.items()):
        summarize_class(predicted_class, member_indices)

    return result
