import numpy as np
import seaborn as sns
from matplotlib.axes import Axes


def plot_kernel_dist(kernel: np.ndarray, ax: Axes):
    low_tri = np.tril(kernel)
    lst = low_tri.flatten()
    sns.histplot(lst, ax=ax)
    # return ax


def plot_kernel_heat(kernel: np.ndarray, ax: Axes):
    mask = np.tril(kernel)  # here has to be a mask because not flattening
    cmap = sns.diverging_palette(230, 20, as_cmap=True)  # later change
    sns.heatmap(
        kernel,
        mask=mask,
        ax=ax,
        cmap=cmap,
        vmax=0.3,
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.5},
    )
