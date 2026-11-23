from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from loguru import logger
from matplotlib.axes import Axes

from nvml._study.config.config import MakeConfig
from nvml.flearn.cluster.interfaces import ClusterData
from nvml.flearn.graphs.labels import LabelSchemeNames
from nvml.flearn.graphs.viz_graphs import plot_one
from nvml.flearn.kernel.interfaces import KernelNames


# TODO: check that kernels are all different
def plot_cluster_group(
    cfg: MakeConfig, cd: ClusterData, indices: list[int], ax_group: list[Axes]
):
    for ax, ix in zip(ax_group, indices):
        case_name = cfg.get_one_case(ix)
        score = cd.coeffs[ix]
        plot_one(cfg, ix, ax, label=f"{case_name} | {score:.2f}")


def plot_similar_graphs(
    cfg: MakeConfig,
    cluster_path: Path,
    kernel: KernelNames,
    scheme: LabelSchemeNames,
    k: int,
):
    cd = ClusterData(cluster_path, kernel, scheme, k)
    cname = cd.name

    logger.info(f"Plotting for {cname}")
    cnts = pl.Series(cd.labels).value_counts(sort=True)
    logger.info(cnts)

    n_to_plot = 4
    fig, axes = plt.subplots(nrows=k, ncols=n_to_plot)

    for cluster_group, ax_group in enumerate(axes):
        logger.info(f"Plotting group {cluster_group}")
        indices = np.flatnonzero(cd.labels == cluster_group)[:n_to_plot].tolist()
        ax_group[0].set_ylabel(f"Group {cluster_group}")
        plot_cluster_group(cfg, cd, indices, ax_group)

    fig.suptitle(f"{cname} | Sil Score: {np.median(cd.coeffs):.2f}")
    print("\n")
    return fig

    #   nested for-loop to plot them..
    # potential to use different labeling shchemes for plotting.. ipx can help here..

    # also report stas: n of cd.labels in each group, kernel_value ..
