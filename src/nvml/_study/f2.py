import matplotlib
import matplotlib.pyplot as plt  # noqa
import numpy as np  # noqa
import polars as pl  # noqa
import polars.selectors as cs  # noqa
import seaborn as sns  # noqa
import xarray as xr
from cyclopts import App
from plan2eplus.ezcase.ez import EZ
from ptpython.repl import embed

from nvml import DataPaths as dp
from nvml import dn  # noqa
from nvml._study.config.cstore import CONFIGS_DICT
from nvml.fexp.make_fexp.gather import gather_constant_data, gather_wind_dependent_data
from nvml.io import get_ambient_ds_from_cfg, read_graph
from nvml.rprep.collate import (
    RegDataPaths,
    prepare_qoi_df,
)
from nvml.rprep.df_names import dfc  # noqa
from nvml.rprep.mlist import (
    make_core_feature_ds,
    make_pca_dfs,
    prepare_many_models,
)

matplotlib.use("module://matplotlib-backend-kitty")
f2_app = App("f2")


cfg = CONFIGS_DICT["msd_aug20"]
REG = dp.reg.v3


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
    n = 10
    # drns: WindDirectionBinNames = ["N", "E", "W", "S", "NW"]  # pyright: ignore[reportAssignmentType]

    drns = ["N", "E", "W", "S"]
    qoi_df = prepare_qoi_df(REG.flow_qoi_dataset)
    rdp = RegDataPaths(REG.const_features, REG.wdep_features)
    pca_dfs = make_pca_dfs(dp.kern.v2.pca_h5, n)
    cdfs = make_core_feature_ds(rdp, drns)
    rd, dfs = prepare_many_models(qoi_df, pca_dfs, cdfs, n, drns)

    # embed(globals(), locals(), vi_mode=True)
    #
    #


@f2_app.command()
def fg1():
    roi_df = xr.open_dataset(REG.flow_qoi_dataset)
    ads = get_ambient_ds_from_cfg(cfg)

    embed(globals(), locals(), vi_mode=True)
