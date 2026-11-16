import seaborn as sns  # noqa
import matplotlib
import xarray as xr  # noqa
import matplotlib.pyplot as plt  # noqa

from nvml import dn  # noqa
from nvml._study.buoy import TEST_ROOM, cfg, TEST_CASE
from nvml.io import read_graph


matplotlib.use("module://matplotlib-backend-kitty")


def get_graph():
    graph = read_graph(cfg.make_graph_path(TEST_CASE))
    assert graph
    return graph


RM = TEST_ROOM
drn = "NORTH"
fg = get_graph()


# graph, da, da_diff = fb()
