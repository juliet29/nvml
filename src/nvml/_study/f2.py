from cyclopts import App
from plan2eplus.ezcase.ez import EZ
from ptpython.repl import embed

from nvml._study.config.cstore import CONFIGS_DICT
from nvml.io import read_graph
from nvml.regression.data.single import gather_constant_data, gather_wind_dependent_data

f2_app = App("f2")


cfg = CONFIGS_DICT["msd_aug20"]


@f2_app.command()
def ff():
    G = read_graph(cfg.get_one_graph_path(0))
    case = EZ(cfg.get_one_case_data().idf)
    assert G
    df = gather_wind_dependent_data(case, G, "hi")
    cd = gather_constant_data(G, "hi")

    embed(globals(), locals(), vi_mode=True)
