from cyclopts import App
from icecream import ic
from loguru import logger

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
    logger.debug(StudyPaths.data.gnn)
    return FlowGraphDataset(cfg, StudyPaths.data.gnn)


@gmod.command()
def fda():
    gds = FlowGraphDataset(cfg, StudyPaths.data.gnn)  # test process
    gds.cluster("N", 2)
    [i for i in gds]
    ic(gds.num_features)

    ic(gds.num_classes)
    # data = gds[0]
    #

    # gds.transform(data)
