import shutil
from pathlib import Path

import torch
from loguru import logger
from torch_geometric.data import Dataset

from nvml.cli.config import MakeConfig
from nvml.gmodel.processing import GModelNames, Processor

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


class FlowGraphDataset(Dataset):
    def __init__(self, cfg: MakeConfig, save_loc: Path):
        self.cfg = cfg
        self.case_names = self.get_case_names()
        super().__init__(root=str(save_loc))

    @property
    def raw_file_names(self):
        logger.debug(self.raw_dir)
        return list(self.cfg.case_names)

    @property
    def processed_file_names(self):
        return [GModelNames.make_processed_data_name(i) for i in self.case_names]

    def get_case_names(self):
        return sorted(self.cfg.case_names)

    def download(self):
        shutil.copytree(self.cfg.data_store, self.raw_dir, dirs_exist_ok=True)

    def process(self):
        pr = Processor(self.cfg, Path(self.processed_dir), self.case_names)
        pr.write_all()

    def len(self):
        return len(self.case_names)

    def get(self, idx: int):
        path = self.processed_paths[idx]
        return torch.load(path, weights_only=False)
