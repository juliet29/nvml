from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import to_rgba
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from nvml.explain.interfaces import ClassExplanation, InstanceExplanation, MotifPrevalence
from nvml.gmodel.hetero import CARDINAL_DIRECTIONS, EXTERNAL, ZONE
from nvml.style import BEIGE, CARDINAL, DUSTY_TAUPE, INK, MAUVE_BARK
from nvml.utils import save_mpl_fig

CARDINAL_COLOR = MAUVE_BARK
ZONE_COLOR = CARDINAL
EDGE_COLOR = DUSTY_TAUPE
NODE_OUTLINE = INK
CARDINAL_SIZE = 420
ZONE_SIZE = 700
EAST_WEST_LEAN = 1.5
ZONE_X_SPACING = 2.6
DEPTH_Y_SPACING = 2.0
STAR_ANCHOR_RADIUS = 3.0
STAR_ZONE_RADIUS = 1.5
STAR_ZONE_SPREAD = 1.3
GALLERY_TITLE_FONTSIZE = 26
GALLERY_SUPTITLE_FONTSIZE = 32

CARDINAL_UNIT_VECTOR = {
    "NORTH": (0.0, 1.0),
    "SOUTH": (0.0, -1.0),
    "EAST": (-1.0, 0.0),
    "WEST": (1.0, 0.0),
}


def to_compass_display_graph(motif: nx.Graph) -> nx.Graph:
    display = nx.Graph()
    display.graph["area_bins"] = motif.graph.get("area_bins")
    for direction in CARDINAL_DIRECTIONS:
        display.add_node(direction, kind="cardinal", direction=direction)

    def anchor_of(node):
        data = motif.nodes[node]
        if data["label"].startswith(EXTERNAL):
            return data["label"].split(":")[1]
        return node

    def add_zone(node):
        data = motif.nodes[node]
        display.add_node(
            node,
            kind="zone",
            label=data["label"],
            area_norm=data.get("area_norm"),
            area_bin=data.get("area_bin"),
        )

    for node, data in motif.nodes(data=True):
        if not data["label"].startswith(EXTERNAL):
            add_zone(node)

    for u, v in motif.edges:
        display.add_edge(anchor_of(u), anchor_of(v))

    return display


def compass_layout(display: nx.Graph) -> dict:
    zone_nodes = [node for node, data in display.nodes(data=True) if data["kind"] == "zone"]
    depth = nx.single_source_shortest_path_length(display, "NORTH") if display.degree("NORTH") else {}
    reachable = [depth[node] for node in zone_nodes if node in depth]
    fallback_depth = max(reachable, default=0) + 1

    members_by_depth: dict[int, list] = defaultdict(list)
    for node in zone_nodes:
        members_by_depth[depth.get(node, fallback_depth)].append(node)

    positions: dict = {}

    def lean_of(node) -> float:
        offset = 0.0
        if display.has_edge(node, "EAST"):
            offset -= EAST_WEST_LEAN
        if display.has_edge(node, "WEST"):
            offset += EAST_WEST_LEAN
        return offset

    def place_layer(node_depth: int, layer: list):
        ordered = sorted(layer, key=lambda node: (lean_of(node), display.nodes[node]["label"]))
        count = len(ordered)
        for slot, node in enumerate(ordered):
            x = (slot - (count - 1) / 2) * ZONE_X_SPACING + lean_of(node)
            positions[node] = (x, -float(node_depth) * DEPTH_Y_SPACING)

    for node_depth, layer in members_by_depth.items():
        place_layer(node_depth, layer)

    bottom = -float(fallback_depth + 1) * DEPTH_Y_SPACING
    half_width = ZONE_X_SPACING + max((abs(x) for x, _ in positions.values()), default=0.0)
    positions["NORTH"] = (0.0, DEPTH_Y_SPACING)
    positions["SOUTH"] = (0.0, bottom)
    positions["EAST"] = (-half_width - 1.0, bottom / 2)
    positions["WEST"] = (half_width + 1.0, bottom / 2)
    return positions


def compass_star_layout(display: nx.Graph) -> dict:
    zone_nodes = [node for node, data in display.nodes(data=True) if data["kind"] == "zone"]
    positions: dict = {}

    def cardinals_of(node) -> tuple:
        return tuple(c for c in CARDINAL_DIRECTIONS if display.has_edge(node, c))

    groups: dict[tuple, list] = defaultdict(list)
    for node in zone_nodes:
        groups[cardinals_of(node)].append(node)

    def place_group(cardinals: tuple, members: list):
        anchor_vectors = [CARDINAL_UNIT_VECTOR[c] for c in cardinals]
        direction = np.mean(anchor_vectors, axis=0) if anchor_vectors else np.zeros(2)
        norm = float(np.linalg.norm(direction))
        unit = direction / norm if norm > 0 else np.zeros(2)
        perpendicular = np.array([-unit[1], unit[0]]) if norm > 0 else np.array([1.0, 0.0])
        base = unit * STAR_ZONE_RADIUS
        count = len(members)
        ordered = sorted(members, key=lambda node: display.nodes[node]["label"])
        for slot, node in enumerate(ordered):
            offset = (slot - (count - 1) / 2) * STAR_ZONE_SPREAD
            point = base + perpendicular * offset
            positions[node] = (float(point[0]), float(point[1]))

    for cardinals, members in groups.items():
        place_group(cardinals, members)

    for direction, (x, y) in CARDINAL_UNIT_VECTOR.items():
        positions[direction] = (x * STAR_ANCHOR_RADIUS, y * STAR_ANCHOR_RADIUS)
    return positions


def draw_typed_subgraph(
    graph: nx.Graph,
    ax: Axes,
    title: str,
    node_scale: float = 1.0,
    layout=compass_layout,
    label_zones: bool = True,
):
    if graph.number_of_nodes() == 0:
        ax.set_title(f"{title}\n(no important subgraph)")
        ax.set_axis_off()
        return

    display = to_compass_display_graph(graph)
    positions = layout(display)

    def short_label_of(node) -> str:
        data = display.nodes[node]
        if data["kind"] == "cardinal":
            return data["direction"][0]
        label = data["label"]
        if label == ZONE:
            return "Z"
        if label.startswith(f"{ZONE}:"):
            return label.replace(f"{ZONE}:", "Z:")
        return label

    def size_of(node) -> float:
        base = CARDINAL_SIZE if display.nodes[node]["kind"] == "cardinal" else ZONE_SIZE
        return base * node_scale

    def color_of(node):
        is_cardinal = display.nodes[node]["kind"] == "cardinal"
        return to_rgba(CARDINAL_COLOR if is_cardinal else ZONE_COLOR)

    nodelist = list(display.nodes)
    short_labels = {
        node: short_label_of(node)
        for node in nodelist
        if label_zones or display.nodes[node]["kind"] == "cardinal"
    }
    node_sizes = [size_of(node) for node in nodelist]
    node_colors = [color_of(node) for node in nodelist]

    nx.draw_networkx_edges(display, positions, ax=ax, edge_color=EDGE_COLOR)
    nx.draw_networkx_nodes(
        display,
        positions,
        nodelist=nodelist,
        node_size=node_sizes,
        node_color=node_colors,
        edgecolors=NODE_OUTLINE,
        ax=ax,
    )
    nx.draw_networkx_labels(
        display, positions, labels=short_labels, font_size=round(7 * node_scale), ax=ax
    )

    ax.set_title(title)
    ax.set_axis_off()
    ax.margins(0.15)


@dataclass
class ClassExplanationReport:
    class_explanation: ClassExplanation
    model_label: str = ""

    def titled(self, text: str) -> str:
        return f"{self.model_label} — {text}" if self.model_label else text

    def plot_directions(self, ax: Axes):
        ce = self.class_explanation
        importance = [ce.direction_importance.get(name, 0.0) for name in CARDINAL_DIRECTIONS]
        coverage = [ce.direction_coverage.get(name, 0.0) for name in CARDINAL_DIRECTIONS]
        bars = ax.bar(CARDINAL_DIRECTIONS, importance, color=CARDINAL)

        def annotate(bar, fraction: float):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{fraction:.0%}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

        for bar, fraction in zip(bars, coverage):
            annotate(bar, fraction)
        ax.set_ylabel("mean normalized importance")
        ax.set_title("external direction (label = coverage)")

    def plot_relations(self, ax: Axes):
        ce = self.class_explanation
        relations = list(ce.relation_importance.keys())
        labels = [f"{src}→{dst}\n{rel}" for src, rel, dst in relations]
        values = [ce.relation_importance[relation] for relation in relations]
        ax.bar(range(len(relations)), values, color=DUSTY_TAUPE)
        ax.set_xticks(range(len(relations)))
        ax.set_xticklabels(labels, fontsize=7)
        ax.set_ylabel("mean normalized edge importance")
        ax.set_title("relation importance")

    def plot_zone_area(self, ax: Axes):
        pairs = self.class_explanation.zone_area_importance
        if not pairs:
            ax.set_title("zone area vs importance\n(no zones)")
            ax.set_axis_off()
            return
        areas = [area for area, _ in pairs]
        importances = [importance for _, importance in pairs]
        ax.scatter(areas, importances, alpha=0.4, s=15, color=MAUVE_BARK)
        ax.set_xlabel("zone area")
        ax.set_ylabel("normalized importance")
        ax.set_title("zone area vs importance")

    def make_motif_gallery(self, top_k: int = 4) -> Figure:
        ce = self.class_explanation
        motifs = ce.motifs[:top_k]
        columns = max(len(motifs), 1)
        fig, axes = plt.subplots(nrows=1, ncols=columns, figsize=(6 * columns, 8))
        axes = axes if isinstance(axes, (list, tuple)) or hasattr(axes, "__len__") else [axes]

        def draw_ranked(rank: int, motif: MotifPrevalence):
            title = (
                f"#{rank + 1}  freq {motif.frequency:.0%}  excl {motif.exclusivity:.0%}\n"
                f"lift {motif.lift:.2f}  (case {motif.representative_case})"
            )
            draw_typed_subgraph(motif.representative_graph, axes[rank], title)

        if not motifs:
            axes[0].set_title("no recurring motifs")
            axes[0].set_axis_off()
        else:
            for rank, motif in enumerate(motifs):
                draw_ranked(rank, motif)
        for ax in axes:
            ax.title.set_fontsize(GALLERY_TITLE_FONTSIZE)

        fig.suptitle(
            self.titled(
                f"class {ce.predicted_class} — recurring key subgraphs "
                f"({ce.n_instances} graphs)"
            ),
            fontsize=GALLERY_SUPTITLE_FONTSIZE,
        )
        fig.tight_layout()

        def draw_panel_divider(left: Axes, right: Axes):
            x = (left.get_position().x1 + right.get_position().x0) / 2
            fig.add_artist(
                Line2D([x, x], [0.05, 0.9], transform=fig.transFigure, color=DUSTY_TAUPE, linewidth=1)
            )

        for left, right in zip(axes[:-1], axes[1:]):
            draw_panel_divider(left, right)
        return fig

    def plot_pathway_multiplicity(self, ax: Axes):
        ce = self.class_explanation
        n = ce.n_instances or 1

        def fraction(counts, keep) -> float:
            return sum(1 for c in counts if keep(c)) / n

        none_fraction, single_fraction, multiple_fraction = [], [], []
        for direction in CARDINAL_DIRECTIONS:
            counts = ce.pathway_counts.get(direction, [])
            none_fraction.append(fraction(counts, lambda c: c == 0))
            single_fraction.append(fraction(counts, lambda c: c == 1))
            multiple_fraction.append(fraction(counts, lambda c: c >= 2))

        positions = range(len(CARDINAL_DIRECTIONS))
        single_bottom = none_fraction
        multiple_bottom = [n_ + s for n_, s in zip(none_fraction, single_fraction)]
        ax.bar(positions, none_fraction, color=BEIGE, edgecolor=INK, linewidth=0.5, label="none")
        ax.bar(positions, single_fraction, bottom=single_bottom, color=DUSTY_TAUPE, edgecolor=INK, linewidth=0.5, label="single")
        ax.bar(positions, multiple_fraction, bottom=multiple_bottom, color=CARDINAL, edgecolor=INK, linewidth=0.5, label="multiple")
        ax.set_xticks(list(positions))
        ax.set_xticklabels(CARDINAL_DIRECTIONS, fontsize=7)
        ax.set_ylim(0, 1)
        ax.set_ylabel("fraction of graphs")
        ax.set_title("flow pathways per cardinal (zones connected)")
        ax.legend(fontsize=7, loc="upper right")

    def make_profile(self) -> Figure:
        ce = self.class_explanation
        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 5))
        self.plot_directions(axes[0])
        self.plot_pathway_multiplicity(axes[1])
        self.plot_zone_area(axes[2])
        fig.suptitle(
            self.titled(f"class {ce.predicted_class} — profile ({ce.n_instances} graphs)")
        )
        fig.tight_layout()
        return fig

    def make_summary(self) -> Figure:
        ce = self.class_explanation
        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 5))
        self.plot_directions(axes[0])
        self.plot_relations(axes[1])
        self.plot_zone_area(axes[2])
        fig.suptitle(
            self.titled(
                f"class {ce.predicted_class} — importance summary ({ce.n_instances} graphs)"
            )
        )
        fig.tight_layout()
        return fig


def make_instance_subgraph(instance: InstanceExplanation) -> Figure:
    fig, ax = plt.subplots(figsize=(6, 6))
    title = f"{instance.case_name} → class {instance.predicted_class}"
    draw_typed_subgraph(instance.motif_graph, ax, title)
    fig.tight_layout()
    return fig


def save_class_reports(
    class_explanations: dict[int, ClassExplanation],
    figures_dir: Path,
    model_label: str = "",
    tag: str = "",
) -> list[Path]:
    explanation_dir = figures_dir / "explanation" / tag
    if explanation_dir.exists():
        for stale in explanation_dir.glob("*.png"):
            stale.unlink()
    saved: list[Path] = []

    def save_class(predicted_class: int, class_explanation: ClassExplanation):
        report = ClassExplanationReport(class_explanation, model_label)
        gallery = report.make_motif_gallery()
        saved.append(
            save_mpl_fig(gallery, explanation_dir / f"class_{predicted_class}_subgraphs.png")
        )
        plt.close(gallery)
        profile = report.make_profile()
        saved.append(
            save_mpl_fig(profile, explanation_dir / f"class_{predicted_class}_profile.png")
        )
        plt.close(profile)

    for predicted_class, class_explanation in sorted(class_explanations.items()):
        save_class(predicted_class, class_explanation)
    return saved


def save_motif_galleries(
    class_explanations: dict[int, ClassExplanation],
    out_dir: Path,
    model_label: str = "",
) -> list[Path]:
    if out_dir.exists():
        for stale in out_dir.glob("*.png"):
            stale.unlink()
    saved: list[Path] = []

    def save_class(predicted_class: int, class_explanation: ClassExplanation):
        gallery = ClassExplanationReport(class_explanation, model_label).make_motif_gallery()
        saved.append(save_mpl_fig(gallery, out_dir / f"class_{predicted_class}_subgraphs.png"))
        plt.close(gallery)

    for predicted_class, class_explanation in sorted(class_explanations.items()):
        save_class(predicted_class, class_explanation)
    return saved
