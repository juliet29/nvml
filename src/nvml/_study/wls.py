import iplotx as ipx
import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
from grakel import WeisfeilerLehman, graph_from_networkx
from icecream import ic
from matplotlib.axes import Axes
from plyze.flow_graph.interfaces import FlowGraph

from nvml._study.config.config import MakeConfig

matplotlib.use("module://matplotlib-backend-kitty")
from cyclopts import App

from nvml._study.config.cstore import CONFIGS_DICT
from nvml.io import read_graph

wls_app = App("wls")


cfg = CONFIGS_DICT["msd_aug20"]


def wrap_subplots(num_plots: int, max_cols: int = 4):
    def division_with_remainder(a: int, b: int):
        q = a // b
        r = a % b
        return q, r

    n_rows, remainder = division_with_remainder(num_plots, max_cols)

    fig, axes = plt.subplots(nrows=n_rows + 1, ncols=max_cols)
    return fig, axes.flatten()


def plot_many_graphs(cfg: MakeConfig, indices: list[int]):
    def plot_one(ix: int, ax: Axes):
        G = read_graph(cfg.get_one_graph_path(ix))
        case_name = cfg.get_one_case(ix)

        ic(ix)

        try:
            res = ipx.plot(G, ax=ax, layout=G.layout)
        except AttributeError as e:
            raise ValueError(
                f"Invalid data for case {case_name}: {G.nodes}, {G.edges}, {G.layout}: {e}"
            )
        ax.set_title(case_name)
        return res[0]

    fig, axes = wrap_subplots(len(indices))
    ic(axes)
    # ic(axes.flatten())
    # breakpoint()

    for ax, ix in zip(axes, indices):
        plot_one(ix, ax)

    plt.show()


@wls_app.command()
def fa():
    G = read_graph(cfg.get_one_graph_path(0))
    fig, ax = plt.subplots()
    res = ipx.plot(G, ax=ax, layout=G.layout)
    res[0]
    plt.show()


@wls_app.command()
def fb():
    plot_many_graphs(cfg, list(range(10)))


@wls_app.command()
def fc():
    LABEL = "label"

    def label_graph(G: FlowGraph):
        # d = {i: ix for ix, i in enumerate(list(G.nodes))}
        d = {i: 1 for ix, i in enumerate(list(G.nodes))}
        nx.set_node_attributes(G, d, LABEL)
        return G

    # convert to grakel
    flow_graphs = [read_graph(cfg.get_one_graph_path(i)) for i in range(10)]
    labeled_graphs = [label_graph(i) for i in flow_graphs]
    gks = list(
        graph_from_networkx(labeled_graphs, as_Graph=True, node_labels_tag=LABEL)
    )

    # for ix, i in enumerate(gks):
    #     ic(ix)
    #     try:
    #         _ = i.get_labels()
    #     except:
    #         raise Exception
    #
    # sp = ShortestPath(normalize=True)
    sp = WeisfeilerLehman(normalize=True, n_iter=2)
    a, b, c, *_ = gks

    # aa = sp.fit_transform([a, a])
    # ab = sp.fit_transform([a, b])
    abc = sp.fit_transform([a, b, c])
    kern = sp.fit_transform(gks[0:4])
    ic(kern)
