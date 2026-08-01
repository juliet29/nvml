from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import torch
from loguru import logger
from plyze import make_flow_graph
from plyze.flow_graph.create.main import make_flow_graph
from plyze.flow_graph.interfaces import FlowGraph
from torch_geometric.utils import from_networkx
from utils4plans.io import make_dir

from nvml.cli.config import MakeConfig
from nvml.cluster.setup.multi import init_zarr, write_to_zarr
from nvml.cluster.spectral import make_spectral
from nvml.cluster.tsne import setup_for_clustering
from nvml.io import get_ambient_data_as_ds


class GModelNames(NamedTuple):
    zarr_name: str = "../cluster/{FileNames.zarr}"

    def make_processed_data_name(self, case_name: str):
        return f"data_{case_name}.pt"


@dataclass
class Processor:
    cfg: MakeConfig
    save_loc: Path
    case_names: list[str]

    @property
    def sql_path(self):
        return self.cfg.get_one_case_data().sql

    @property
    def zarr_path(self):
        return self.save_loc / GModelNames.zarr_name

    @property
    def ambient_ds(self):
        return get_ambient_data_as_ds(self.sql_path)

    @property
    def graphs(self):
        def load(case_name):
            return make_flow_graph(
                self.cfg.make_case_data(case_name), self.cfg.cardinal_expansion_factor
            )

        return [load(i) for i in self.case_names]

    def init_zarr(self):
        init_zarr(self.save_loc, self.case_names)

    def write_one_to_zarr(self, graph: FlowGraph, case_name: str):
        write_to_zarr(
            case_name,
            self.zarr_path,
            graph=graph,
            ambient_ds_or_sql_path=self.ambient_ds,
        )

    def write_one_to_torch(self, graph: FlowGraph, case_name: str):
        torch_data = from_networkx(graph)
        make_dir(self.save_loc)
        torch.save(
            torch_data, self.save_loc / GModelNames(case_name).make_processed_data_name
        )

    def write_all(self):
        self.init_zarr()
        for ix, (g, c) in enumerate(zip(self.graphs, self.case_names)):
            logger.info(f"Processing {c} ({ix + 1}/{len(self.case_names)})")
            self.write_one_to_torch(g, c)
            # TODO: check -> dont graphs have a case name property?
            self.write_one_to_zarr(g, c)


# this is pre-load.. doesnt get saved to the graph. -> transform / pre-transform
def cluster(
    zarr_path: Path, wind_sector: str, n_clusters: int = 2, random_state: int = 1204
):  # TODO: make wind_sector an enum
    da = setup_for_clustering(zarr_path)
    model = make_spectral(da, wind_sector, n_clusters, random_state)
    pass
