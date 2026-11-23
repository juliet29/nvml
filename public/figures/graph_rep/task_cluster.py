from pathlib import Path

import matplotlib.pyplot as plt
from loguru import logger
from pytask import task
from utils4plots.save import save_mpl_fig

from nvml import DataPaths
from nvml._study.config.cstore import CONFIGS_DICT
from nvml.flearn.cluster.interfaces import ClusterId
from nvml.flearn.cluster.viz_clusters import plot_similar_graphs
from nvml.settings import TaskProduct, settings

LOC = "cluster"
cfg = CONFIGS_DICT["msd_aug20"]

BIG_SIZE = (15, 8.5)  # TODO: somewhat plot specific

studies: list[ClusterId] = [
    ClusterId("sm", "int_ext", 4),
    ClusterId("sm", "int_ext", 2),
    ClusterId("sp", "area_ratio", 3),
    ClusterId("sp", "int_ext", 2),
    ClusterId("wl", "int_ext", 2),
    ClusterId("wl", "int_ext", 3),
]

for ix, group in enumerate(studies):

    @task(id=str(ix), kwargs={"group": group})
    def task_plot_graph_clusters(
        group: ClusterId,
        cluster_path: Path = DataPaths.kern.v2.clusters_h5,
        out_path: TaskProduct = settings.fig_path(
            LOC, f"{group.kernel}_{group.scheme}_{group.k}"
        ),
    ):
        logger.info(group)

        with plt.rc_context({"figure.figsize": BIG_SIZE}):
            fig = plot_similar_graphs(
                cfg, cluster_path, group.kernel, group.scheme, group.k
            )
            save_mpl_fig(fig, out_path)
