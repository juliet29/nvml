from pathlib import Path

import pytask

from nvml import DataPaths
from nvml._study.config.cstore import CONFIGS_DICT
from nvml.settings import TaskProduct
from nvml.wl.cluster_data import save_clusters
from nvml.wl.graphs.gk_dataset import ManyLabelsGklDatasetMaker
from nvml.wl.kernel.kern_data import save_kernels
from nvml.wl.pca.pca_data import save_pcas

if None:

    @pytask.mark.persist(reason="Already generated and renamed manually")
    def task_make_gk_graphs(
        out: TaskProduct = DataPaths.kern.v2.gk_graphs_joblib(test=False),
    ):

        cfg = CONFIGS_DICT[
            "msd_aug20"
        ]  # should go to a config likely / should also be an explicit input
        n_jobs = 10

        gkdm = ManyLabelsGklDatasetMaker(
            cfg, dir_path=out, true_run=True, save=True, njobs=n_jobs
        )
        gkdm.run()

    @pytask.mark.persist(reason="Already generated")
    def task_make_gk_kernel(
        gk_graph_path: Path = DataPaths.kern.v2.gk_graphs_joblib(test=False),
        out: TaskProduct = DataPaths.kern.v2.kernel_matrices_h5,
    ):
        save_kernels(gk_graph_path, out)

    @pytask.mark.persist(reason="Already generated")
    def task_make_gk_clusters(
        kernel_path: Path = DataPaths.kern.v2.kernel_matrices_h5,
        out: TaskProduct = DataPaths.kern.v2.clusters_h5,
    ):
        save_clusters(kernel_path, out)


@pytask.mark.persist(reason="Already generated")
def task_make_gk_pca(
    kernel_path: Path = DataPaths.kern.v2.kernel_matrices_h5,
    out: TaskProduct = DataPaths.kern.v2.pca_h5,
):
    save_pcas(kernel_path, out)
