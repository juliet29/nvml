from pathlib import Path

import iplotx as ipx
import matplotlib
import matplotlib.pyplot as plt
from grakel import ShortestPath, VertexHistogram, WeisfeilerLehman  # noqa

from nvml.wl.kernel.prep import GklDatasetMaker
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
def fe():
    p = Path("")
    fn = "fake.csv"
    rng = (0, 30)
    gkdm = GklDatasetMaker(cfg, p, fn, test_rng=rng)
    gkdm.run()
    kern = WeisfeilerLehman(normalize=True).fit(gkdm.gk_graphs)
    return gkdm.gk_graphs, kern  # graphs


@wls_app.command()
def ff():
    # test fancy labels and kernel
    n_graphs = 5
    flow_graphs = [read_graph(cfg.get_one_graph_path(i)) for i in range(n_graphs)]
    return flow_graphs
