from cyclopts import App
from plan2eplus.ezcase.ez import EZ
from ptpython.repl import embed

from nvml import DataPaths
from nvml._study.config.cstore import CONFIGS_DICT
from nvml.io import read_graph
from nvml.regression.data.gather import gather_constant_data, gather_wind_dependent_data
from nvml.regression.models.collate import (
    RegDataId,
    RegDataPaths,
    collate_data,
    prepare_qoi_df,
)

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


@f2_app.command()
def fg():
    rdi = RegDataId("N", "wl", "int_ext", 5)
    dp = DataPaths
    reg = dp.reg.v3
    rdp = RegDataPaths(reg.const_features, reg.wdep_features, dp.kern.v2.pca_h5)
    qoi_df = prepare_qoi_df(reg.flow_qoi_dataset)
    df = collate_data(rdp, rdi, qoi_df)

    embed(globals(), locals(), vi_mode=True)
