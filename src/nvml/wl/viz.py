import iplotx as ipx
import matplotlib
import matplotlib.pyplot as plt
from icecream import ic
from matplotlib.axes import Axes
from plyze.flow_graph.interfaces import FlowGraph

from nvml._study.config.config import MakeConfig

matplotlib.use("module://matplotlib-backend-kitty")

from nvml.io import read_graph


def plot_graph(G: FlowGraph):
    return G.layout
    pass


def wrap_subplots(num_plots: int, max_cols: int = 4):
    def division_with_remainder(a: int, b: int):
        q = a // b
        r = a % b
        return q, r

    n_rows, remainder = division_with_remainder(num_plots, max_cols)

    fig, axes = plt.subplots(nrows=n_rows + 1, ncols=max_cols)
    return fig, axes.flatten()


def plot_one(cfg: MakeConfig, ix: int, ax: Axes, label: str = ""):
    G = read_graph(cfg.get_one_graph_path(ix))
    assert G

    case_name = cfg.get_one_case(ix)

    if not label:
        label = case_name

    try:
        res = ipx.plot(G, ax=ax, layout=G.layout)
    except AttributeError as e:
        raise ValueError(
            f"Invalid data for case {case_name}: {G.nodes}, {G.edges}, {G.layout}: {e}"
        )
    ax.set_title(label)
    return res[0]


def plot_many_graphs(cfg: MakeConfig, indices: list[int]):
    fig, axes = wrap_subplots(len(indices))
    ic(axes)
    # ic(axes.flatten())
    # breakpoint()

    for ax, ix in zip(axes, indices):
        plot_one(cfg, ix, ax)

    return fig
