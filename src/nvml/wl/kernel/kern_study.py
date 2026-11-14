from typing import Callable, Literal, get_args

import matplotlib.pyplot as plt
import numpy as np
from grakel import Graph as grk_graph
from grakel import (  # noqa
    GraphletSampling,
    Kernel,
    ShortestPath,
    SubgraphMatching,
    VertexHistogram,
    WeisfeilerLehman,
)
from icecream import ic
from matplotlib.axes import Axes

from nvml.wl.kernel.labels import LabelSchemeNames
from nvml.wl.kernel.viz import plot_kernel_dist, plot_kernel_heat
from nvml.wl.sil import score_plot, sil_dist_plot, spc, study_sil_score

KFxType = Callable[[list[grk_graph]], np.ndarray]
KernelNames = Literal["wl", "sp", "sm"]


def mkf(grk_class: type[Kernel]):
    return grk_class(normalize=True)


kernel_fxs: dict[KernelNames, Kernel] = {
    "wl": mkf(WeisfeilerLehman),
    "sp": mkf(ShortestPath),
    "sm": mkf(SubgraphMatching),
    # "gs": mkf(GraphletSampling),
}


def try_clustering_and_plot(
    kernel: np.ndarray, ax: Axes, k_to_try: list[int] = [2, 3, 4, 5], n: int = 2
):
    df = study_sil_score(kernel, k_to_try)
    top_k = (
        df.sort(by=spc.score, descending=True)[0:n].get_column(spc.k)
        # .cast(pl.Int32)
        # .to_list()
    )
    score_plot(df, ax)
    ttk = [int(i) for i in top_k]

    return ttk


def make_kernel_and_plot(
    graphs: list[grk_graph],
    axes: list[Axes],
    title: str = "",
    kernel_name: KernelNames = "wl",
) -> np.ndarray:
    kernel = kernel_fxs[kernel_name].fit_transform(graphs)
    ax1, ax2 = axes
    plot_kernel_dist(kernel, ax1)  # pyright: ignore[reportArgumentType]
    plot_kernel_heat(kernel, ax2)  # pyright: ignore[reportArgumentType]
    ax1.set_title(title)
    return kernel  # pyright: ignore[reportReturnType]


def plot_for_all_schemes(
    gd: dict[LabelSchemeNames, list[grk_graph]], kernel_name: KernelNames = "wl"
):
    n_rows = len(gd.keys())

    fig, axes = plt.subplots(n_rows, 5, figsize=(20, 20))

    for key, ax_group in zip(gd.keys(), axes):
        ax1, ax2, ax3, ax4, ax5 = ax_group
        kernel = make_kernel_and_plot(
            gd[key],
            (ax1, ax2),
            title=key,
            kernel_name=kernel_name,  # pyright: ignore[reportArgumentType]
        )  # pyright: ignore[reportArgumentType]
        top_k = try_clustering_and_plot(kernel, ax3)

        sil_dist_plot(kernel, top_k[0], ax4)
        sil_dist_plot(kernel, top_k[1], ax5)

    return fig


def plots_for_all_kernels(gd: dict[LabelSchemeNames, list[grk_graph]]):
    for name in get_args(KernelNames):
        ic(name)
        fig = plot_for_all_schemes(gd, name)
        fig.tight_layout()
        fig.suptitle(name)
        plt.show()
