from grakel import VertexHistogram, WeisfeilerLehman, ShortestPath  # noqa
import matplotlib.pyplot as plt
from grakel import Graph as grk_graph
from matplotlib.axes import Axes

from nvml.wl.kernel.labels import LabelSchemeNames
from nvml.wl.kernel.viz import plot_kernel_dist, plot_kernel_heat


def make_kernel_and_plot(graphs: list[grk_graph], axes: list[Axes], title: str = ""):
    kernel = WeisfeilerLehman(normalize=True).fit_transform(graphs)
    ax1, ax2 = axes
    plot_kernel_dist(kernel, ax1)  # pyright: ignore[reportArgumentType]
    plot_kernel_heat(kernel, ax2)  # pyright: ignore[reportArgumentType]
    ax1.set_title(title)
    return


def plot_for_all_schemes(gd: dict[LabelSchemeNames, list[grk_graph]]):
    n_rows = len(gd.keys())

    fig, axes = plt.subplots(n_rows, 2, figsize=(10, 12))
    for key, ax_pair in zip(gd.keys(), axes):
        make_kernel_and_plot(gd[key], ax_pair, title=key)

    return fig
