from pathlib import Path
from typing import Any, Callable

import matplotlib
from grakel import Graph as grk_graph
from grakel import graph_from_networkx
from plyze.flow_graph.interfaces import FlowGraph
from utils4plans.computation.loops2 import DatasetMaker

from nvml._study.config.config import MakeConfig
from nvml.io import read_graph
from nvml.wl.kernel.labels import LABEL, label_graph_null

matplotlib.use("module://matplotlib-backend-kitty")


# def convert_graphs():


class GklDatasetMaker(DatasetMaker):
    def __init__(
        self,
        cfg: MakeConfig,
        dir_path: Path,
        file_name: str = "",
        label_fx: Callable[[FlowGraph], FlowGraph] = label_graph_null,
        true_run: bool = False,
        save: bool = False,
        njobs: int = 3,
        test_rng: tuple[int, int] = (0, 5),
    ) -> None:
        self.cfg = cfg
        self.label_fx = label_fx
        super().__init__(cfg, dir_path, file_name, true_run, save, njobs, test_rng)

    def process_fx(self, obj: None | Any = None, ix: int = 0):
        path = self.cfg.get_one_graph_path(ix)
        case_name = self.cfg.get_one_case(ix)
        try:
            flow_graph = read_graph(path)
            if not flow_graph:
                return None

            return self.label_fx(flow_graph)
        except Exception as e:
            print(f"Issue proccessing {ix}:{case_name} -- {e}")
            pass

    def agglom_fx(self, labeled_graphs: list[FlowGraph]) -> list[grk_graph]:
        print("Transforming graphs!")
        self.gk_graphs = list(
            graph_from_networkx(labeled_graphs, as_Graph=True, node_labels_tag=LABEL)
        )

        return self.gk_graphs  # pyright: ignore[reportReturnType]

    def save_fx(self, grk_graphs: list[grk_graph]):
        # going to save different kernels

        print("Not saving this!")
        pass
