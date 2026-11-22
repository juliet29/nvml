from pathlib import Path
from typing import Annotated

import matplotlib.pyplot as plt
from loguru import logger
from pytask import Product, task
from utils4plots.save import save_mpl_fig

from nvml import DataPaths
from nvml._study.config.cstore import CONFIGS_DICT
from nvml.settings import settings
from nvml.flearn.pca.make_pca import PCAId
from nvml.flearn.pca.viz_align import plot_component_graphs

LOC = "pca"
cfg = CONFIGS_DICT["msd_aug20"]

BIG_SIZE = (15, 8.5)  # TODO: somewhat plot specific

studies: list[PCAId] = [
    # PCAId("sm", "int_ext"),
    # PCAId("sm", "int_ext"),
    # PCAId("sp", "area_ratio"),
    # PCAId("sp", "int_ext"),
    # PCAId("wl", "discrete_area_ratio"),
    PCAId("wl", "int_ext"),
]

COMPONENT_NUMS = [0, 1, 2]
NUM_SAMPLES = 4
METHOD = "ordered"

for ix, group in enumerate(studies):
    OUTPUT_FILES = [
        settings.fig_path(LOC, group.path_name, METHOD, f"comp_{ix}")
        for ix in COMPONENT_NUMS
    ]

    @task(id=str(ix), kwargs={"group": group})
    def task_plot_graph_pca_components(
        group: PCAId,
        pca_path: Path = DataPaths.kern.v2.pca_h5,
        out_paths: Annotated[list[Path], Product] = OUTPUT_FILES,
    ):
        logger.info(group)

        with plt.rc_context({"figure.figsize": BIG_SIZE}):
            figs = plot_component_graphs(
                cfg, pca_path, group, COMPONENT_NUMS, NUM_SAMPLES, method=METHOD
            )
            logger.info("Saving now")

            for fig, path in zip(figs, out_paths):
                save_mpl_fig(fig, path)
