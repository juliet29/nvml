from pathlib import Path
from typing import Any

import matplotlib
import pandas as pd
from grakel import Graph as grk_graph
from grakel import graph_from_networkx
from plyze.flow_graph.interfaces import FlowGraph
from utils4plans.computation.loops2 import DatasetMaker

from nvml._study.config.config import MakeConfig
from nvml.io import read_graph
from nvml.wl.kernel.labels import LabelSchemeNames, LabelTags, try_all_labels

matplotlib.use("module://matplotlib-backend-kitty")


# def convert_graphs():


class ManyLabelsGklDatasetMaker(DatasetMaker):
    def __init__(
        self,
        cfg: MakeConfig,
        dir_path: Path,
        file_name: str = "",
        true_run: bool = False,
        save: bool = False,
        njobs: int = 3,
        test_rng: tuple[int, int] = (0, 5),
    ) -> None:
        self.cfg = cfg
        super().__init__(cfg, dir_path, file_name, true_run, save, njobs, test_rng)

    def process_fx(self, obj: None | Any = None, ix: int = 0):
        path = self.cfg.get_one_graph_path(ix)
        case_name = self.cfg.get_one_case(ix)
        try:
            flow_graph = read_graph(path)
            if not flow_graph:
                return None

            return try_all_labels(flow_graph)
        except Exception as e:
            print(f"Issue proccessing {ix}:{case_name} -- {e}")

    def agglom_fx(
        self, labeled_graphs: list[dict[LabelSchemeNames, FlowGraph]]
    ) -> pd.DataFrame:
        print("Transforming graphs!")

        def mk_grakel(gs: list[FlowGraph]):
            return list(
                graph_from_networkx(
                    gs,
                    as_Graph=True,
                    node_labels_tag=LabelTags.NODE_LABEL,
                    edge_labels_tag=LabelTags.EDGE_LABEL,
                )
            )

        df = pd.DataFrame(labeled_graphs)  # .to_dict(orient="list")
        self.mod_df = df.apply(mk_grakel)
        assert isinstance(self.mod_df, pd.DataFrame)

        return self.mod_df

    def save_fx(self, grk_graphs: list[grk_graph]):
        # going to save different kernels

        print("Not saving this!")
        pass
