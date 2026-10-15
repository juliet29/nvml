from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from plan2eplus.visuals.base.base_plot import BasePlot
from plyze.flow_graph.interfaces import FlowGraph

from nvml.explain.interfaces import ClassExplanation, MotifPrevalence
from nvml.gmodel.hetero import ZONE
from nvml.style import BEIGE, CARDINAL, INK
from nvml.utils import save_mpl_fig

HIGHLIGHT = CARDINAL
MOTIF_EDGE_WIDTH = 6
MOTIF_EDGE_HALO_WIDTH = 11
MOTIF_NODE_SIZE = 420
MOTIF_NODE_OUTLINE_WIDTH = 2.6
HALO_COLOR = BEIGE
NODE_OUTLINE_COLOR = INK


@dataclass
class FloorPlanMotifOverlay:
    load_graph: Callable[[str], FlowGraph]
    make_plan_plot: Callable[[str], BasePlot]

    def draw_over_motif(
        self, motif: MotifPrevalence, predicted_class: int, title: str | None = None
    ) -> Figure:
        base = self.make_plan_plot(motif.representative_case)
        graph = self.load_graph(motif.representative_case)
        subgraph = motif.representative_graph

        def location_of(motif_key) -> tuple[float, float]:
            node_type, local_index = motif_key
            nodes = graph.zone_nodes if node_type == ZONE else graph.external_nodes
            return tuple(nodes[local_index].data.location.as_tuple)

        def highlight_edge(source_key, target_key):
            (source_x, source_y) = location_of(source_key)
            (target_x, target_y) = location_of(target_key)
            (line,) = base.axes.plot(
                [source_x, target_x],
                [source_y, target_y],
                color=HIGHLIGHT,
                linewidth=MOTIF_EDGE_WIDTH,
                solid_capstyle="round",
                zorder=5,
            )
            line.set_path_effects(
                [
                    path_effects.Stroke(
                        linewidth=MOTIF_EDGE_HALO_WIDTH, foreground=HALO_COLOR
                    ),
                    path_effects.Normal(),
                ]
            )

        def highlight_node(motif_key):
            x, y = location_of(motif_key)
            dots = base.axes.scatter(
                [x],
                [y],
                s=MOTIF_NODE_SIZE,
                color=HIGHLIGHT,
                edgecolors=NODE_OUTLINE_COLOR,
                linewidths=MOTIF_NODE_OUTLINE_WIDTH,
                zorder=6,
            )
            dots.set_path_effects(
                [
                    path_effects.Stroke(
                        linewidth=MOTIF_EDGE_HALO_WIDTH, foreground=HALO_COLOR
                    ),
                    path_effects.Normal(),
                ]
            )

        for source_key, target_key in subgraph.edges:
            highlight_edge(source_key, target_key)
        for motif_key in subgraph.nodes:
            highlight_node(motif_key)

        default_title = (
            f"{motif.representative_case} — class {predicted_class} "
            f"(lift {motif.lift:.2f}, {motif.frequency:.0%})"
        )
        base.axes.set_title(title or default_title)
        base.finalize()
        return base.fig


def save_motif_overlays(
    class_explanations: dict[int, ClassExplanation],
    overlay: FloorPlanMotifOverlay,
    figures_dir: Path,
    tag: str = "",
    top_k: int = 4,
) -> list[Path]:
    overlay_dir = figures_dir / "explanation" / tag / "overlays"
    if overlay_dir.exists():
        for stale in overlay_dir.glob("*.png"):
            stale.unlink()
    saved: list[Path] = []

    def save_motif(predicted_class: int, rank: int, motif: MotifPrevalence):
        fig = overlay.draw_over_motif(motif, predicted_class)
        name = f"class_{predicted_class}_motif_{rank + 1}_{motif.representative_case}.png"
        saved.append(save_mpl_fig(fig, overlay_dir / name))
        plt.close(fig)

    def save_class(predicted_class: int, class_explanation: ClassExplanation):
        for rank, motif in enumerate(class_explanation.motifs[:top_k]):
            save_motif(predicted_class, rank, motif)

    for predicted_class, class_explanation in sorted(class_explanations.items()):
        save_class(predicted_class, class_explanation)
    return saved
