from cyclopts import App

from nvml import DataPaths
from nvml._study.config.cstore import CONFIGS_DICT
from nvml.fexp.make_fexp.rrun_data import RegDatasetMaker
from nvml.settings import TaskProduct

TRUE_RUN, SAVE = False, True
N_JOBS = 3

# TODO: cyclopts for data, pytask for figures.

features = App("fe")


@features.command()
def make(
    true_run: bool = TRUE_RUN,
    save: bool = SAVE,
    njobs: int = N_JOBS,
    wdep: TaskProduct = DataPaths.reg.v3.wdep_features,
    const: TaskProduct = DataPaths.reg.v3.const_features,
):

    cfg = CONFIGS_DICT["msd_aug20"]
    fq = RegDatasetMaker(
        cfg, wdep_path=wdep, const_path=const, true_run=true_run, save=save, njobs=njobs
    )
    fq.run()
