from pathlib import Path
from typing import Annotated

import pytask
from pytask import Product

from nvml import DataPaths
from nvml._study.config.cstore import CONFIGS_DICT
from nvml.wl.dataset.gk_dataset import ManyLabelsGklDatasetMaker
from nvml.wl.kernel.kern_data import save_kernels


@pytask.mark.persist(reason="Already generated and renamed manually")
def task_make_gk_graphs(
    out: Annotated[Path, Product] = DataPaths.kern.v2.gk_graphs_joblib(test=False),
):

    cfg = CONFIGS_DICT["msd_aug20"]  # should go to a config likely
    n_jobs = 10

    gkdm = ManyLabelsGklDatasetMaker(
        cfg, dir_path=out, true_run=True, save=True, njobs=n_jobs
    )
    gkdm.run()


def task_make_gk_kernel(
    gk_graph_path: Path = DataPaths.kern.v2.gk_graphs_joblib(test=False),
    out: Annotated[Path, Product] = DataPaths.kern.v2.kernel_matrices_h5,
):
    save_kernels(gk_graph_path, out)
