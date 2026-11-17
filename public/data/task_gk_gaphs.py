from pathlib import Path
from typing import Annotated

from pytask import Product

from nvml import DataPaths
from nvml._study.config.cstore import CONFIGS_DICT
from nvml.wl.kernel.gk_dataset import ManyLabelsGklDatasetMaker


def task_make_gk_graphs(
    out: Annotated[Path, Product] = DataPaths.kern.v2.gk_graphs_parquet(test=False),
):

    cfg = CONFIGS_DICT["msd_aug20"]  # should go to a config likely
    n_jobs = 10

    gkdm = ManyLabelsGklDatasetMaker(
        cfg, dir_path=out, true_run=True, save=True, njobs=n_jobs
    )
    gkdm.run()
