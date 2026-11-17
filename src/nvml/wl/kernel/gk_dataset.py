from pathlib import Path
from typing import Any

import joblib
import matplotlib
import pandas as pd
from grakel import Graph as grk_graph
from grakel import graph_from_networkx
from loguru import logger
from plyze.flow_graph.interfaces import FlowGraph
from utils4plans.computation.loops2 import DatasetMaker

from nvml._study.config.config import MakeConfig
from nvml.io import read_graph
from nvml.wl.kernel.labels import LabelSchemeNames, LabelTags, try_all_labels

matplotlib.use("module://matplotlib-backend-kitty")


def mk_grakel(fg: FlowGraph) -> grk_graph:
    g = list(
        graph_from_networkx(
            [fg],
            as_Graph=True,
            node_labels_tag=LabelTags.NODE_LABEL,
            edge_labels_tag=LabelTags.EDGE_LABEL,
        )
    )[0]
    assert isinstance(g, grk_graph)
    return g


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

            label_dict = try_all_labels(flow_graph)
            gk_label_dict = {k: mk_grakel(v) for k, v in label_dict.items()}
            return gk_label_dict

        except Exception as e:
            logger.critical(f"Issue proccessing {ix}:{case_name} -- {e}")

    def agglom_fx(self, labeled_graphs: list[dict[LabelSchemeNames, grk_graph]]):
        logger.info("Agglomerating data!")

        self.graph_dict: dict[LabelSchemeNames, list[grk_graph]] = pd.DataFrame(
            labeled_graphs
        ).to_dict(orient="list")
        return self.graph_dict

    def save_fx(self, gd: dict):
        joblib.dump(self.graph_dict, self.save_path)
        logger.success("Saved graphs!")
