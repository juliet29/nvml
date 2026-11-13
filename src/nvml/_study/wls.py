import iplotx as ipx
import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
from grakel import WeisfeilerLehman, graph_from_networkx
from icecream import ic
from plyze.flow_graph.interfaces import FlowGraph

from nvml.wl.viz import plot_many_graphs

matplotlib.use("module://matplotlib-backend-kitty")
from cyclopts import App

from nvml._study.config.cstore import CONFIGS_DICT
from nvml.io import read_graph

wls_app = App("wls")


cfg = CONFIGS_DICT["msd_aug20"]


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
    plt.show()


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

    sp = WeisfeilerLehman(normalize=True, n_iter=2)
    a, b, c, *_ = gks

    kern = sp.fit_transform(gks[0:4])
    ic(kern)
    return kern, sp, gks


def try_classify():
    pass
