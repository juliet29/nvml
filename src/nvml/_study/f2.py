import matplotlib
import matplotlib.pyplot as plt  # noqa
import numpy as np  # noqa
import polars as pl  # noqa
import polars.selectors as cs  # noqa
import seaborn as sns  # noqa
import xarray as xr
from cyclopts import App
from ptpython.repl import embed

from nvml import DataPaths as dp
from nvml import dn  # noqa
from nvml._study.config.cstore import CONFIGS_DICT
from nvml.io import get_ambient_ds_from_cfg, read_graph
from nvml.qoi.defns import (
    calc_flow_qois,
    calc_interzonal_ratio,
    calc_uniformity_index,
    plan_qdim,
)
from nvml.rprep.collate import (
    RegDataPaths,
    prepare_qoi_df,
)
from nvml.nvml.df_names import dfc  # noqa
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
    # case = EZ(cfg.get_one_case_data().idf)
    assert G
    U = get_ambient_ds_from_cfg(cfg)[dn.wind_speed]
    ads = get_ambient_ds_from_cfg(cfg)

    return calc_flow_qois(G, ads)
    return calc_uniformity_index(G, U)
    da = plan_qdim(G, U)
    # plan_da = calc_zone_inflow_across_plan(G, U)
    return calc_interzonal_ratio(da)

    # df = gather_wind_dependent_data(case, G, "hi")
    # cd = gather_constant_data(G, "hi")

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
    ds = xr.open_dataset(REG.flow_qoi_dataset)
    ads = get_ambient_ds_from_cfg(cfg)

    embed(globals(), locals(), vi_mode=True)
