from nvml import DataPaths
from nvml._study.config.cstore import CONFIGS_DICT
from nvml.flow_qoi.make_flow_data import FlowQOIDatasetMaker
from nvml.settings import TaskProduct


def task_make_flow_qoi(
    out: TaskProduct = DataPaths.reg.v3.flow_qoi_dataset,
):

    cfg = CONFIGS_DICT["msd_aug20"]
    n_jobs = 12

    fq = FlowQOIDatasetMaker(cfg, dir_path=out, true_run=True, save=True, njobs=n_jobs)
    fq.run()
