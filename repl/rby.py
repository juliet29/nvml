import matplotlib
import matplotlib.pyplot as plt  # noqa
import pandas as pd  # noqa
import seaborn as sns  # noqa
import xarray as xr  # noqa
from plyze.qoi.xarray_helpers import CardinalNameLiteral

from nvml import dn  # noqa
from nvml._study.buoy import TEST_CASE, TEST_ROOM, cfg
from nvml.qoi_studies.buoy.data import BuoyData
from nvml.io import read_graph

matplotlib.use("module://matplotlib-backend-kitty")


def get_graph():
    graph = read_graph(cfg.make_graph_path(TEST_CASE))
    assert graph
    return graph


RM = TEST_ROOM
drn: CardinalNameLiteral = "NORTH"
# fg = get_graph()


def make_bd():
    bd = BuoyData(cfg, TEST_CASE, RM, drn)
    bd.calc_ri()
    bd.calc_dim_flow_rate()

    df = bd.init_df().with_columns(
        **{
            dn.q_dim: bd.q_dim.values,
            dn.t_dim: bd.t_dim.values,
            dn.T: bd.T_ext.values,
            dn.U: bd.U.values,
            dn.Ri: bd.Ri.values,
        }
    )
    return bd, df


bd = make_bd()


# graph, da, da_diff = fb()
