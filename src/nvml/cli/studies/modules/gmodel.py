from cyclopts import App

from nvml.cli.config import CONFIGS_DICT
from nvml.cli.studies.helpers import cfg
from nvml.cli.studies.study_paths import StudyPaths
from nvml.gmodel.dataset import FlowGraphDataset
from nvml.gmodel.model_utils import prep_data

gmod = App("gmod")


@gmod.command()
def fc():
    pass
    # return graph_to_torch_data(cfg, CASE_NAME, StudyPaths.data.gnn)


@gmod.command()
def fd():
    # test dataset
    return FlowGraphDataset(cfg, StudyPaths.data.gnn.test)


@gmod.command()
def fe():
    # larger test dataset
    return FlowGraphDataset(cfg, StudyPaths.data.gnn.case50)


@gmod.command()
def ff():
    cfg = CONFIGS_DICT["case50"]
    p = StudyPaths.data.gnn.case50
    # gds = FlowGraphDataset(cfg,   # test process
    sdl = prep_data(cfg, p)
    return sdl
