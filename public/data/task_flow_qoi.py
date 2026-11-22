from cyclopts import App

from nvml import DataPaths
from nvml._study.config.cstore import CONFIGS_DICT
from nvml.qoi.make_flow_data import FlowQOIDatasetMaker
from nvml.settings import TaskProduct

TRUE_RUN, SAVE = False, True
N_JOBS = 3


qois = App("fq")


@qois.command()
def make(
    true_run: bool = TRUE_RUN,
    save: bool = SAVE,
    njobs: int = N_JOBS,
    out: TaskProduct = DataPaths.reg.v3.flow_qoi_dataset,
):

    cfg = CONFIGS_DICT["msd_aug20"]
    fq = FlowQOIDatasetMaker(
        cfg, dir_path=out, true_run=true_run, save=save, njobs=njobs
    )
    fq.run()
