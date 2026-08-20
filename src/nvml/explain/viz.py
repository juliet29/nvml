from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from nvml.explain.interfaces import ClassExplanation, InstanceExplanation, MotifPrevalence
from nvml.gmodel.hetero import CARDINAL_DIRECTIONS, EXTERNAL
from nvml.utils import save_mpl_fig

MOTIF_LAYOUT_SEED = 1204


def draw_typed_subgraph(graph: nx.Graph, ax: Axes, title: str):
    if graph.number_of_nodes() == 0:
        ax.set_title(f"{title}\n(no important subgraph)")
        ax.set_axis_off()
        return
    labels = nx.get_node_attributes(graph, "label")
    node_colors = [
        "tab:orange" if labels[node].startswith(EXTERNAL) else "tab:blue"
        for node in graph
    ]
    short_labels = {node: labels[node].replace(f"{EXTERNAL}:", "") for node in graph}
    positions = nx.spring_layout(graph, seed=MOTIF_LAYOUT_SEED)
    nx.draw_networkx(
        graph,
        positions,
        ax=ax,
        node_color=node_colors,
        labels=short_labels,
        font_size=7,
        node_size=450,
        edge_color="0.6",
    )
    ax.set_title(title)
    ax.set_axis_off()


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
        bars = ax.bar(CARDINAL_DIRECTIONS, importance, color="tab:blue")

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
        ax.bar(range(len(relations)), values, color="tab:green")
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
        ax.scatter(areas, importances, alpha=0.4, s=15, color="tab:blue")
        ax.set_xlabel("zone area")
        ax.set_ylabel("normalized importance")
        ax.set_title("zone area vs importance")

    def make_motif_gallery(self, top_k: int = 4) -> Figure:
        ce = self.class_explanation
        motifs = ce.motifs[:top_k]
        columns = max(len(motifs), 1)
        fig, axes = plt.subplots(nrows=1, ncols=columns, figsize=(5 * columns, 5))
        axes = axes if isinstance(axes, (list, tuple)) or hasattr(axes, "__len__") else [axes]

        def draw_ranked(rank: int, motif: MotifPrevalence):
            title = (
                f"#{rank + 1}  count {motif.count}  lift {motif.lift:.2f}\n"
                f"freq {motif.frequency:.0%}"
            )
            draw_typed_subgraph(motif.representative_graph, axes[rank], title)

        if not motifs:
            axes[0].set_title("no recurring motifs")
            axes[0].set_axis_off()
        else:
            for rank, motif in enumerate(motifs):
                draw_ranked(rank, motif)

        fig.suptitle(
            self.titled(
                f"class {ce.predicted_class} — recurring key subgraphs "
                f"({ce.n_instances} graphs)"
            )
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
    explanation_dir = figures_dir / "explanation"
    suffix = f"_{tag}" if tag else ""
    saved: list[Path] = []

    def save_class(predicted_class: int, class_explanation: ClassExplanation):
        report = ClassExplanationReport(class_explanation, model_label)
        gallery = report.make_motif_gallery()
        saved.append(
            save_mpl_fig(
                gallery,
                explanation_dir / f"gnn_explanation{suffix}_class_{predicted_class}_subgraphs.png",
            )
        )
        plt.close(gallery)
        summary = report.make_summary()
        saved.append(
            save_mpl_fig(
                summary,
                explanation_dir / f"gnn_explanation{suffix}_class_{predicted_class}_summary.png",
            )
        )
        plt.close(summary)

    for predicted_class, class_explanation in sorted(class_explanations.items()):
        save_class(predicted_class, class_explanation)
    return saved
