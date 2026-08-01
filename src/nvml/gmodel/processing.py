from dataclasses import dataclass
from pathlib import Path

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
from nvml.constants import DataNames, FileNames
from nvml.io import get_ambient_data_as_ds
from nvml.qdim.wind import WindDirectionBinNames


class GModelNames:
    @classmethod
    def make_cluster_path(cls, save_loc: Path):
        p = save_loc.parent / "cluster"
        make_dir(p)
        return p

    # @classmethod
    # def make_zarr_name(cls, save_loc: Path):
    #     p = cls.make_cluster_path(save_loc)
    # return p / f"{FileNames.zarr}"

    @classmethod
    def make_processed_data_name(cls, case_name: str):
        return f"data_{case_name}.pt"

    @classmethod
    def make_cluster_name(
        cls, cluster_path: Path, wind_direction: WindDirectionBinNames, n_clusters: int
    ):

        p = cluster_path
        assert cluster_path.exists()
        return p / f"spectral_model_{wind_direction}_{n_clusters}.nc"


@dataclass
class Processor:
    cfg: MakeConfig
    save_loc: Path
    case_names: list[str]

    @property
    def sql_path(self):
        return self.cfg.get_one_case_data().sql

    @property
    def cluster_path(self):
        return GModelNames.make_cluster_path(self.save_loc)

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
        init_zarr(self.cluster_path, self.case_names)

    def write_one_to_zarr(self, graph: FlowGraph, case_name: str):
        write_to_zarr(
            case_name,
            self.cluster_path / FileNames.zarr,
            graph=graph,
            ambient_ds_or_sql_path=self.ambient_ds,
        )

    def write_one_to_torch(self, graph: FlowGraph, case_name: str):
        torch_data = from_networkx(graph)
        torch_data[DataNames.case_name] = case_name
        torch.save(
            torch_data,
            self.save_loc / GModelNames.make_processed_data_name(case_name),
        )

    def write_all(self):
        self.init_zarr()
        for ix, (g, c) in enumerate(zip(self.graphs, self.case_names)):
            logger.info(f"Processing {c} ({ix + 1}/{len(self.case_names)})")
            self.write_one_to_torch(g, c)
            # TODO: check -> dont graphs have a case name property?
            self.write_one_to_zarr(g, c)


# this is pre-load.. doesnt get saved to the graph. -> transform / pre-transform
def save_spectal_clusters(
    save_loc: Path,
    wind_sector: WindDirectionBinNames,
    n_clusters: int = 2,
    random_state: int = 1204,
):  # TODO: make wind_sector an enum
    cluster_path = GModelNames.make_cluster_path(save_loc)
    zarr_path = cluster_path / FileNames.zarr

    da = setup_for_clustering(zarr_path)
    model = make_spectral(da, wind_sector, n_clusters, random_state)

    path = GModelNames.make_cluster_name(cluster_path, wind_sector, n_clusters)
    # TODO: use a named tuple for hyperparams, save in a metadata folder..
    model.to_netcdf(path)
    return path
