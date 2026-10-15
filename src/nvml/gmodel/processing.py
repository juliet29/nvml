from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import torch
from loguru import logger
from plyze import FlowGraphModel
from plyze.flow_graph.create.main import make_flow_graph
from plyze.flow_graph.interfaces import FlowGraph
from rich.pretty import pretty_repr
from tqdm import tqdm
from utils4plans.io import make_dir

from nvml.cli.config.config import MakeConfig
from nvml.cluster.setup.multi import init_zarr, write_to_zarr
from nvml.cluster.spectral import make_spectral
from nvml.cluster.tsne import setup_for_clustering
from nvml.constants import DataNames, FileNames
from nvml.gmodel.dataset_interfaces import ClusterModelParams
from nvml.gmodel.hetero import flow_graph_to_hetero_data
from nvml.logs import silence_plan2eplus_logging
from nvml.qdim.wind import WindDirectionBinNames
from nvml.utils import default_worker_count


class GModelNames:
    @classmethod
    def make_cluster_path(cls, save_loc: Path):
        p = save_loc.parent / "cluster"
        make_dir(p)
        return p

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

    @classmethod
    def make_model_name(cls, tag: str):
        return f"gnn_{tag}.pt"

    @classmethod
    def make_best_model_name(cls, tag: str):
        return f"gnn_best_{tag}.pt"


def load_flow_graph(cfg: MakeConfig, case_name: str, reuse_existing: bool) -> FlowGraph:
    json_path = cfg.make_graph_path(case_name)
    if reuse_existing and json_path.exists():
        return FlowGraphModel.read(json_path)
    return make_flow_graph(cfg.make_case_data(case_name), cfg.cardinal_expansion_factor)


MIN_ZONES_FOR_CLUSTERING = 5


def build_case(
    cfg: MakeConfig,
    save_loc: Path,
    zarr_path: Path,
    sql_path: Path,
    case_name: str,
    reuse_existing: bool,
):
    graph = load_flow_graph(cfg, case_name, reuse_existing)
    if len(graph.zone_nodes) < MIN_ZONES_FOR_CLUSTERING:
        return None
    write_to_zarr(case_name, zarr_path, graph=graph, ambient_ds_or_sql_path=sql_path)
    data = flow_graph_to_hetero_data(graph, case_name)
    torch.save(data, save_loc / GModelNames.make_processed_data_name(case_name))
    return case_name


@dataclass
class Processor:
    cfg: MakeConfig
    save_loc: Path
    case_names: list[str]
    reuse_existing: bool = True
    workers: int | None = None

    @property
    def sql_path(self):
        return self.cfg.get_one_case_data().sql

    @property
    def cluster_path(self):
        return GModelNames.make_cluster_path(self.save_loc)

    def torch_path(self, case_name: str):
        return self.save_loc / GModelNames.make_processed_data_name(case_name)

    def build_all(self):
        init_zarr(self.cluster_path, self.case_names)
        zarr_path = self.cluster_path / FileNames.zarr
        sql_path = self.sql_path

        failures = []
        results = []
        max_workers = self.workers if self.workers is not None else default_worker_count()
        with ProcessPoolExecutor(
            max_workers=max_workers, initializer=silence_plan2eplus_logging
        ) as executor:
            case_by_future = {
                executor.submit(
                    build_case,
                    self.cfg,
                    self.save_loc,
                    zarr_path,
                    sql_path,
                    case,
                    self.reuse_existing,
                ): case
                for case in self.case_names
            }

            def record(completed: Future[str | None]):
                case = case_by_future[completed]
                try:
                    results.append(completed.result())
                except Exception as e:
                    failures.append({case: f"{e}"})

            for completed in tqdm(
                as_completed(case_by_future),
                total=len(case_by_future),
                desc="processing",
            ):
                record(completed)

        dropped = results.count(None)
        if dropped:
            logger.info(f"Dropped {dropped} cases with < {MIN_ZONES_FOR_CLUSTERING} zones")
        if failures:
            logger.error(f"Failed cases summary: {pretty_repr(failures)}")


def save_spectal_clusters(
    save_loc: Path,
    model_params: ClusterModelParams,
    random_state: int = 1204,
):
    wind_sector = model_params.wind_sector
    n_clusters = model_params.n_clusters
    cluster_path = GModelNames.make_cluster_path(save_loc)
    zarr_path = cluster_path / FileNames.zarr

    da = setup_for_clustering(zarr_path)
    retained_cases = da[DataNames.case_name].values.tolist()
    model = make_spectral(da, wind_sector, n_clusters, random_state)

    path = GModelNames.make_cluster_name(cluster_path, wind_sector, n_clusters)
    model.to_netcdf(path)
    return path, retained_cases
