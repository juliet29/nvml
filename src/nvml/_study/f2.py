from cyclopts import App
from plan2eplus.ezcase.ez import EZ
from ptpython.repl import embed

from nvml._study.config.cstore import CONFIGS_DICT
from nvml.io import read_graph
from nvml.regression.features.coinc import get_coinc_for_all_normal_drns
from nvml.regression.features.dom_node import calc_degree_dominant_node_all_drns

f2_app = App("f2")


cfg = CONFIGS_DICT["msd_aug20"]


@f2_app.command()
def ff():
    G = read_graph(cfg.get_one_graph_path(0))
    case = EZ(cfg.get_one_case_data().idf)
    assert G
    dn = calc_degree_dominant_node_all_drns(G)
    co = get_coinc_for_all_normal_drns(G, case)

    embed(globals(), locals(), vi_mode=True)
