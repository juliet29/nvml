from collections import Counter

import networkx as nx
from rich.console import Console
from rich.table import Table

from nvml.explain.interfaces import ClassExplanation, MotifPrevalence

TOP_MOTIFS = 10


def motif_makeup(graph: nx.Graph) -> str:
    labels = [graph.nodes[node]["label"] for node in graph.nodes]
    counts = Counter(labels)
    parts = [f"{count}x {label}" for label, count in sorted(counts.items())]
    return ", ".join(parts)


def report_top_motifs(
    class_explanations: dict[int, ClassExplanation], top_n: int = TOP_MOTIFS
):
    console = Console()

    def report_class(predicted_class: int, class_explanation: ClassExplanation):
        n_instances = class_explanation.n_instances
        table = Table(
            title=f"class {predicted_class} — top {top_n} motifs (n={n_instances})"
        )
        table.add_column("motif")
        table.add_column("occurs", justify="right")
        table.add_column("freq", justify="right")
        table.add_column("excl", justify="right")
        table.add_column("lift", justify="right")
        table.add_column("makeup")

        def add_row(rank: int, motif: MotifPrevalence):
            name = f"{predicted_class}-{chr(ord('A') + rank)}"
            table.add_row(
                name,
                f"{motif.count}/{n_instances}",
                f"{motif.frequency:.0%}",
                f"{motif.exclusivity:.0%}",
                f"{motif.lift:.2f}",
                motif_makeup(motif.representative_graph),
            )

        for rank, motif in enumerate(class_explanation.motifs[:top_n]):
            add_row(rank, motif)
        console.print(table)

    for predicted_class, class_explanation in sorted(class_explanations.items()):
        report_class(predicted_class, class_explanation)
