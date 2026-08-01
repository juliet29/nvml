import shutil
from pathlib import Path

import torch
import xarray as xr
from loguru import logger
from torch_geometric.data import Data, Dataset

from nvml.cli.config import MakeConfig
from nvml.constants import DataNames
from nvml.gmodel.processing import GModelNames, Processor, save_spectal_clusters
from nvml.qdim.wind import WindDirectionBinNames

# TODO: create a dataset that matches with the pytorch_geometric interface based on plyze.
# will probably have multiple datasets based on the different models want to run
#
# need to: move temp data over to scratch, creaate a project config so remove scratch name from repo..
# read in the graphs as flow graphs
# associate each graph with a label based on current algo
# trim down to the data needed..
#
#
# research: best way to compare different graph data vs compare different models vs compare different hyperparams..
# finish reading explainer epaper which probably details how can know if exolanation is good


def assign_cluster_label_to_data(data: Data, path_to_clustering_model: Path | None):
    # ic(f"Being trasnsformed | before {data}")
    if not path_to_clustering_model:
        raise Exception("Need to cluster!")
    da = xr.open_dataarray(path_to_clustering_model)
    case_name = data[DataNames.case_name]
    try:
        class_idx = da.sel({DataNames.case_name: case_name}).data.reshape(1)
    except KeyError:
        logger.error(f"{case_name} not in da.case_names")
        breakpoint()
    # ic(class_idx)
    data[DataNames.torch_geometric_graph_label] = torch.tensor(
        class_idx, dtype=torch.long
    )
    # ic(f"After: {data}")
    return data


class FlowGraphDataset(Dataset):
    def __init__(self, cfg: MakeConfig, save_loc: Path, force_reload: bool = False):
        logger.info(f"Initializing data that will be saved to {save_loc}")
        self.cfg = cfg
        self.case_names = self.get_case_names()
        self.cluster_path: Path | None = None

        def transform_fx(data: Data):
            data = assign_cluster_label_to_data(
                data, path_to_clustering_model=self.cluster_path
            )
            return data

        super().__init__(
            root=str(save_loc), force_reload=force_reload, transform=transform_fx
        )

    def get_case_names(self):
        return sorted(set(self.cfg.case_names))

    @property
    def raw_file_names(self):
        return list(set(self.case_names))

    @property
    def processed_file_names(self):
        return [GModelNames.make_processed_data_name(i) for i in self.case_names]

    def download(self):
        logger.info(f"`Downloading` data from {self.cfg.data_store}")
        shutil.copytree(self.cfg.data_store, self.raw_dir, dirs_exist_ok=True)

    def process(self):
        logger.info(f"Processing data into {self.processed_dir}")
        # breakpoint()
        pr = Processor(self.cfg, Path(self.processed_dir), self.case_names)
        pr.write_all()

    def cluster(self, wind_sector: WindDirectionBinNames, n_clusters: int):
        path, retained_case_names = save_spectal_clusters(
            Path(self.processed_dir), wind_sector, n_clusters
        )
        # update case names.
        self.cluster_path = path
        self.case_names = sorted(retained_case_names)

    def len(self):
        return len(self.case_names)

    def get(self, idx: int):
        path = self.processed_paths[idx]
        return torch.load(path, weights_only=False)
