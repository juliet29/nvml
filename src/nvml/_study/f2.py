from cyclopts import App
from plan2eplus.ezcase.ez import EZ
from plan2eplus.geometry.directions import WallNormal
from ptpython.repl import embed

from nvml._study.config.cstore import CONFIGS_DICT
from nvml.io import read_graph
from nvml.regression.features.coinc import calc_plan_coinc

f2_app = App("f2")


cfg = CONFIGS_DICT["msd_aug20"]


@f2_app.command()
def ff():
    G = read_graph(cfg.get_one_graph_path(0))
    case = EZ(cfg.get_one_case_data().idf)
    assert G
    da = calc_plan_coinc(G, case, WallNormal.NORTH)
    embed(globals(), locals(), vi_mode=True)
