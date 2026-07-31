import shutil
from pathlib import Path
from typing import NamedTuple

import torch
from loguru import logger
from plyze.flow_graph.create.main import make_flow_graph
from torch_geometric.data import Dataset
from torch_geometric.utils import from_networkx
from utils4plans.io import make_dir

from nvml.cli.config import MakeConfig

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


class GModelNames(NamedTuple):
    case_name: str

    @property
    def processed_data(self):
        return f"data_{self.case_name}.pt"


# from path to pytorch data
def graph_to_torch_data(cfg: MakeConfig, case_name: str, save_loc: Path):
    G = make_flow_graph(cfg.make_case_data(case_name), cfg.cardinal_expansion_factor)
    torch_data = from_networkx(G)
    # TODO: add SOME info about a label.. can do dummy for now. => based on number of nodes..
    make_dir(save_loc)
    torch.save(torch_data, save_loc / GModelNames(case_name).processed_data)
    return torch_data


class FlowGraphDataset(Dataset):
    def __init__(self, cfg: MakeConfig, save_loc: Path):

        # TODO: try to use more of their domain language, ie "root" => look at the docs
        self.cfg = cfg
        # self.root = save_loc
        self.make_case_name_map()

        super().__init__(root=str(save_loc))
        # will be looking for raw_data in the raw_data_dir, so update the config datastore
        # self.cfg.data_store = self.raw_data_dir

    @property
    def raw_file_names(self):
        logger.debug(self.raw_dir)
        return list(self.cfg.case_names)

    @property
    def processed_file_names(self):
        return [str(GModelNames(i).processed_data) for i in self.case_name_map.values()]

    def download(self):
        # copy over the contents of the folder, redirect the cfg (later..)
        # TODO: make sure these have matching names to what is needed..
        shutil.copytree(self.cfg.data_store, self.raw_dir, dirs_exist_ok=True)

    def make_case_name_map(self):
        sorted_case_names = sorted(self.cfg.case_names)
        d = {ix: name for ix, name in enumerate(sorted_case_names)}
        self.case_name_map = d

    def process(self):
        # to keep idx => case_name map constant, sort first
        for case_name in self.case_name_map.values():
            graph_to_torch_data(
                self.cfg, case_name, Path(self.processed_dir)
            )  # TODO: fix

    def len(self):
        return len(self.case_name_map.values())

    def get(self, idx: int):
        # case_name = self.case_name_map[idx]
        path = self.processed_paths[idx]
        # path = self.root / GModelNames(case_name).processed_data
        data = torch.load(path)
        return data
