from cyclopts import App

from nvml.cli.studies.helpers import CASE_NAME, cfg
from nvml.cli.studies.study_paths import StudyPaths
from nvml.gmodel.dataset import FlowGraphDataset, graph_to_torch_data

gmod = App("gmod")


@gmod.command()
def fc():
    return graph_to_torch_data(cfg, CASE_NAME, StudyPaths.data.gnn)


@gmod.command()
def fd():
    return FlowGraphDataset(cfg, StudyPaths.data.gnn)
