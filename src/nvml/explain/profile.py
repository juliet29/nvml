from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from nvml.explain.interfaces import MotifPrevalence, RawExplanation
from nvml.explain.overlay import FloorPlanMotifOverlay
from nvml.explain.subgraph import (
    ZONE_FLOW_EXTERNAL,
    normalized_zone_areas,
    zone_bin_letter,
)
from nvml.explain.viz import compass_star_layout, draw_typed_subgraph
from nvml.gmodel.hetero import CARDINAL_DIRECTIONS, EXTERNAL, ZONE
from nvml.style import DUSTY_TAUPE
from nvml.utils import save_mpl_fig

AREA_MISMATCH_PENALTY = 0.5
CANONICAL_NODE_SCALE = 2.6
CANONICAL_TITLE_FONTSIZE = 26
CANONICAL_SUPTITLE_FONTSIZE = 32


@dataclass
class ConnectivityProfile:
    case_name: str
    predicted_class: int
    pathways: list[tuple[int, int]]
    direction_of_external: dict[int, str]
    zones_by_direction: dict[str, list[int]]
    area_bin_of_zone: dict[int, int]


@dataclass
class ClusterSignature:
    predicted_class: int
    n_instances: int
    multiplicity: dict[str, int]
    dominant_bin: dict[str, int]
    n_bins: int
    area_labels: tuple[str, ...] | None
    representative_case: str
    representative_graph: nx.Graph
    representative_distance: float


def zone_type_label(area_bin: int, n_bins: int, area_labels: tuple[str, ...] | None) -> str:
    if n_bins <= 1:
        return ZONE
    return f"{ZONE}:{zone_bin_letter(area_bin, n_bins, area_labels)}"


def extract_connectivity(
    raw: RawExplanation,
    area_cuts: tuple[float, ...] | None = None,
    area_labels: tuple[str, ...] | None = None,
) -> ConnectivityProfile:
    normalized_areas = normalized_zone_areas(raw)
    use_area_bins = bool(area_cuts) and normalized_areas is not None

    def area_bin_of(zone_index: int) -> int:
        if not use_area_bins:
            return 0
        return int(np.digitize(normalized_areas[zone_index], area_cuts))

    area_bin_of_zone = {
        zone_index: area_bin_of(zone_index) for zone_index in range(len(raw.zone_areas))
    }

    pathways = raw.relation_edges.get(ZONE_FLOW_EXTERNAL, [])
    direction_of_external = {
        external_index: direction
        for external_index, direction in enumerate(raw.external_directions)
    }

    zones_by_direction: dict[str, list[int]] = defaultdict(list)
    seen_by_direction: dict[str, set[int]] = defaultdict(set)

    def register_pathway(zone_index: int, external_index: int):
        direction = direction_of_external[external_index]
        if zone_index not in seen_by_direction[direction]:
            seen_by_direction[direction].add(zone_index)
            zones_by_direction[direction].append(zone_index)

    for zone_index, external_index in pathways:
        register_pathway(zone_index, external_index)

    return ConnectivityProfile(
        case_name=raw.case_name,
        predicted_class=raw.predicted_class,
        pathways=pathways,
        direction_of_external=direction_of_external,
        zones_by_direction=dict(zones_by_direction),
        area_bin_of_zone=area_bin_of_zone,
    )


def representative_subgraph(
    profile: ConnectivityProfile, n_bins: int, area_labels: tuple[str, ...] | None
) -> nx.Graph:
    graph = nx.Graph()
    graph.graph["area_bins"] = n_bins if n_bins > 1 else None

    def add_pathway(zone_index: int, external_index: int):
        direction = profile.direction_of_external[external_index]
        zone_key = (ZONE, zone_index)
        external_key = (EXTERNAL, external_index)
        if zone_key not in graph:
            area_bin = profile.area_bin_of_zone[zone_index]
            graph.add_node(
                zone_key,
                label=zone_type_label(area_bin, n_bins, area_labels),
                area_bin=area_bin,
                area_norm=None,
            )
        if external_key not in graph:
            graph.add_node(external_key, label=f"{EXTERNAL}:{direction}")
        graph.add_edge(zone_key, external_key)

    for zone_index, external_index in profile.pathways:
        add_pathway(zone_index, external_index)
    return graph


def canonical_graph(signature: ClusterSignature) -> nx.Graph:
    graph = nx.Graph()
    graph.graph["area_bins"] = signature.n_bins if signature.n_bins > 1 else None
    next_zone_index = 0

    def add_direction(direction: str):
        nonlocal next_zone_index
        count = signature.multiplicity[direction]
        if count == 0:
            return
        external_key = (EXTERNAL, direction)
        graph.add_node(external_key, label=f"{EXTERNAL}:{direction}")
        area_bin = signature.dominant_bin.get(direction, 0)
        label = zone_type_label(area_bin, signature.n_bins, signature.area_labels)
        for _ in range(count):
            zone_key = (ZONE, next_zone_index)
            next_zone_index += 1
            graph.add_node(zone_key, label=label, area_bin=area_bin, area_norm=None)
            graph.add_edge(zone_key, external_key)

    for direction in CARDINAL_DIRECTIONS:
        add_direction(direction)
    return graph


def build_cluster_signatures(
    profiles: list[ConnectivityProfile],
    area_cuts: tuple[float, ...] | None = None,
    area_labels: tuple[str, ...] | None = None,
) -> dict[int, ClusterSignature]:
    n_bins = len(area_cuts) + 1 if area_cuts else 1
    by_class: dict[int, list[ConnectivityProfile]] = defaultdict(list)
    for profile in profiles:
        by_class[profile.predicted_class].append(profile)

    signatures: dict[int, ClusterSignature] = {}

    def register_class(predicted_class: int, members: list[ConnectivityProfile]):
        multiplicity: dict[str, int] = {}
        dominant_bin: dict[str, int] = {}

        def summarize_direction(direction: str):
            counts = [len(member.zones_by_direction.get(direction, [])) for member in members]
            multiplicity[direction] = int(round(float(np.median(counts)))) if counts else 0
            if multiplicity[direction] > 0:
                bins = [
                    member.area_bin_of_zone[zone_index]
                    for member in members
                    for zone_index in member.zones_by_direction.get(direction, [])
                ]
                dominant_bin[direction] = Counter(bins).most_common(1)[0][0] if bins else 0

        for direction in CARDINAL_DIRECTIONS:
            summarize_direction(direction)

        def distance_to_signature(profile: ConnectivityProfile) -> float:
            distance = 0.0
            for direction in CARDINAL_DIRECTIONS:
                actual_zones = profile.zones_by_direction.get(direction, [])
                distance += abs(multiplicity[direction] - len(actual_zones))
                if multiplicity[direction] > 0 and actual_zones:
                    actual_bins = [profile.area_bin_of_zone[zone] for zone in actual_zones]
                    actual_dominant = Counter(actual_bins).most_common(1)[0][0]
                    if actual_dominant != dominant_bin.get(direction, 0):
                        distance += AREA_MISMATCH_PENALTY
            return distance

        closest = min(members, key=distance_to_signature)
        signatures[predicted_class] = ClusterSignature(
            predicted_class=predicted_class,
            n_instances=len(members),
            multiplicity=multiplicity,
            dominant_bin=dominant_bin,
            n_bins=n_bins,
            area_labels=area_labels,
            representative_case=closest.case_name,
            representative_graph=representative_subgraph(closest, n_bins, area_labels),
            representative_distance=distance_to_signature(closest),
        )

    for predicted_class, members in by_class.items():
        register_class(predicted_class, members)
    return signatures


def make_cluster_contrast(
    signatures: dict[int, ClusterSignature], model_label: str = ""
) -> Figure:
    ordered = sorted(signatures.items())
    columns = max(len(ordered), 1)
    fig, axes = plt.subplots(nrows=1, ncols=columns, figsize=(6 * columns, 8))
    axes = axes if hasattr(axes, "__len__") else [axes]

    def draw_cluster(column: int, predicted_class: int, signature: ClusterSignature):
        pathways = ", ".join(
            f"{direction[0]}×{signature.multiplicity[direction]}"
            for direction in CARDINAL_DIRECTIONS
            if signature.multiplicity[direction] > 0
        )
        title = (
            f"cluster {predicted_class}  ({signature.n_instances} graphs)\n"
            f"{pathways or 'no cardinal pathways'}"
        )
        draw_typed_subgraph(
            canonical_graph(signature),
            axes[column],
            title,
            node_scale=CANONICAL_NODE_SCALE,
            layout=compass_star_layout,
            label_zones=False,
        )

    for column, (predicted_class, signature) in enumerate(ordered):
        draw_cluster(column, predicted_class, signature)
    for ax in axes:
        ax.title.set_fontsize(CANONICAL_TITLE_FONTSIZE)

    suffix = " — canonical connectivity per cluster"
    fig.suptitle(
        f"{model_label}{suffix}" if model_label else suffix.lstrip(" —"),
        fontsize=CANONICAL_SUPTITLE_FONTSIZE,
    )
    fig.tight_layout()

    def draw_panel_divider(left, right):
        x = (left.get_position().x1 + right.get_position().x0) / 2
        fig.add_artist(
            Line2D([x, x], [0.05, 0.9], transform=fig.transFigure, color=DUSTY_TAUPE, linewidth=1)
        )

    for left, right in zip(axes[:-1], axes[1:]):
        draw_panel_divider(left, right)
    return fig


def save_cluster_contrast(
    signatures: dict[int, ClusterSignature],
    figures_dir: Path,
    model_label: str = "",
    tag: str = "",
) -> Path:
    contrast = make_cluster_contrast(signatures, model_label)
    path = figures_dir / "explanation" / tag / "cluster_contrast" / "canonical.png"
    saved = save_mpl_fig(contrast, path)
    plt.close(contrast)
    return saved


def save_signature_overlays(
    signatures: dict[int, ClusterSignature],
    overlay: FloorPlanMotifOverlay,
    figures_dir: Path,
    tag: str = "",
) -> list[Path]:
    overlay_dir = figures_dir / "explanation" / tag / "cluster_contrast"
    for stale in overlay_dir.glob("overlay_*.png"):
        stale.unlink()
    saved: list[Path] = []

    def save_overlay(predicted_class: int, signature: ClusterSignature):
        representative = MotifPrevalence(
            motif_hash=str(predicted_class),
            representative_graph=signature.representative_graph,
            representative_case=signature.representative_case,
            count=signature.n_instances,
            frequency=1.0,
            lift=1.0,
            exclusivity=1.0,
        )
        title = (
            f"cluster {predicted_class} representative — {signature.representative_case}"
        )
        fig = overlay.draw_over_motif(representative, predicted_class, title)
        name = f"overlay_cluster_{predicted_class}_{signature.representative_case}.png"
        saved.append(save_mpl_fig(fig, overlay_dir / name))
        plt.close(fig)

    for predicted_class, signature in sorted(signatures.items()):
        save_overlay(predicted_class, signature)
    return saved
