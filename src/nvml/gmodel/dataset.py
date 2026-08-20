from pathlib import Path

import torch
import xarray as xr
from loguru import logger
from torch_geometric.data import HeteroData, InMemoryDataset

from nvml.cli.config.config import MakeConfig
from nvml.constants import DataNames
from nvml.gmodel.dataset_interfaces import ClusterModelParams
from nvml.gmodel.processing import Processor, save_spectal_clusters


def assign_cluster_label_to_data(data: HeteroData, cluster_labels: dict | None):
    if cluster_labels is None:
        raise Exception("Need to cluster!")
    case_name = getattr(data, DataNames.case_name)
    class_idx = cluster_labels[case_name]
    setattr(
        data,
        DataNames.torch_geometric_graph_label,
        torch.tensor([class_idx], dtype=torch.long),
    )
    return data


class FlowGraphDataset(InMemoryDataset):
    def __init__(self, cfg: MakeConfig, save_loc: Path, reuse_existing: bool = True):
        logger.info(f"Initializing data that will be saved to {save_loc}")
        self.cfg = cfg
        self.reuse_existing = reuse_existing
        self.all_case_names = sorted(set(cfg.case_names))
        self.cluster_path: Path | None = None
        self.cluster_labels: dict | None = None
        self.retained_indices: list[int] = []

        def transform_fx(data: HeteroData):
            return assign_cluster_label_to_data(data, self.cluster_labels)

        super().__init__(root=str(save_loc), transform=transform_fx)
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ["data.pt"]

    def process(self):
        pr = Processor(
            self.cfg, Path(self.processed_dir), self.all_case_names, self.reuse_existing
        )
        pr.build_all()

        built_case_names = [c for c in self.all_case_names if pr.torch_path(c).exists()]
        data_list = [
            torch.load(pr.torch_path(c), weights_only=False) for c in built_case_names
        ]
        self.save(data_list, self.processed_paths[0])

    def stored_case_names(self):
        return [getattr(self.get(i), DataNames.case_name) for i in range(len(self))]

    def cluster(self, model_params: ClusterModelParams, seed: int = 1204):
        path, retained_case_names = save_spectal_clusters(
            Path(self.processed_dir), model_params, random_state=seed
        )
        self.cluster_path = path

        da = xr.open_dataarray(path)
        self.cluster_labels = {
            name: int(label)
            for name, label in zip(
                da[DataNames.case_name].values.tolist(), da.values.tolist()
            )
        }

        retained = set(retained_case_names) & set(self.cluster_labels)
        self.retained_indices = [
            ix for ix, name in enumerate(self.stored_case_names()) if name in retained
        ]

    @property
    def num_classes(self):
        assert self.cluster_labels is not None
        return len(set(self.cluster_labels.values()))
