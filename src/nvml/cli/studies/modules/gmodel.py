from cyclopts import App
from icecream import ic

from nvml.cli.config import CONFIGS_DICT
from nvml.cli.studies.helpers import cfg
from nvml.cli.studies.study_paths import StudyPaths
from nvml.gmodel.dataset import FlowGraphDataset

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

    cfg = CONFIGS_DICT["case50"]
    gds = FlowGraphDataset(cfg, StudyPaths.data.gnn.case50)  # test process

    ic(gds.len())
    gds.cluster("N", 3)
    # [i for i in gds]
    ic(gds.num_features)

    ic(gds.num_classes)
    ic(gds.len())
    # data = gds[0]
    #

    # gds.transform(data)
