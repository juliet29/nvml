import matplotlib
from cyclopts import App
from plan2eplus.ezcase.ez import EZ
from ptpython.repl import embed

from nvml import DataPaths as dp
from nvml import dn  # noqa
from nvml._study.config.cstore import CONFIGS_DICT
from nvml._study.utils import *
from nvml.df_names import dfc  # noqa
from nvml.fexp.coinc import get_coinc_for_all_normal_drns
from nvml.fexp.dom_node_all import calc_dom_node_flow_metrics_for_all_drns
from nvml.fexp.make_fexp.gather import gather_wind_dependent_data
from nvml.io import read_graph

matplotlib.use("module://matplotlib-backend-kitty")
fe1_app = App("fe1")


cfg = CONFIGS_DICT["msd_aug20"]
REG = dp.reg.v3


@fe1_app.command()
def ff():
    G = read_graph(cfg.get_one_graph_path(0))
    case = EZ(cfg.get_one_case_data().idf)
    assert G

    df = gather_wind_dependent_data(case, G, "4969")
    return df
    co = get_coinc_for_all_normal_drns(G, case)
    fm = calc_dom_node_flow_metrics_for_all_drns(G)
    embed(globals(), locals(), vi_mode=True)
