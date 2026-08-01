from cyclopts import App
from loguru import logger

from nvml.cli.studies.helpers import cfg
from nvml.cli.studies.study_paths import StudyPaths
from nvml.gmodel.dataset import FlowGraphDataset
from nvml.io import get_ambient_data_as_ds
from nvml.qdim.wind import add_wind_sector_coord

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
    gds = FlowGraphDataset(cfg, StudyPaths.data.gnn, force_reload=True)  # test process
    gds.cluster("N", 2)


@gmod.command()
def fdaa():
    path = cfg.get_one_case_data(0).sql

    ambient_ds = (
        get_ambient_data_as_ds(path).pipe(add_wind_sector_coord)
        # .pipe(wind_sector_as_categorical)
    )
    return ambient_ds
