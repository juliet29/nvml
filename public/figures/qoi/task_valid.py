from pathlib import Path

import matplotlib.pyplot as plt
import xarray as xr
from cyclopts import App
from utils4plans.io import make_dir
from utils4plots import save_mpl_fig

from nvml import DataPaths as dp
from nvml import dn
from nvml._study.config.cstore import CONFIGS_DICT
from nvml.io import get_ambient_ds_from_cfg
from nvml.qoi.viz_amb import (
    plot_ambient_dep_box_per_qoi,
    plot_ambient_dep_scatter_per_qoi,
)
from nvml.qoi.viz_no_corr import plot_no_qoi_corr
from nvml.nvml.df_names import dfc
from nvml.settings import settings

BIG_SIZE = (15, 7.5)  # TODO: move to utils4plots

REG_SRC = dp.reg.v3
cfg = CONFIGS_DICT["msd_aug20"]


fig_qoi_valid = App("fqv")
LOC = "qoi_valid"


@fig_qoi_valid.command()
def amb_scatter(
    qoi: str = dn.intake,
    loc: str = LOC,
    name: str = "amb_scatter",
    flow_qoi: Path = REG_SRC.flow_qoi_dataset,
):
    ads = get_ambient_ds_from_cfg(cfg)
    fq = xr.open_dataset(flow_qoi)

    with plt.rc_context({"figure.figsize": BIG_SIZE}):
        figs, fig_labels = plot_ambient_dep_scatter_per_qoi(qoi, fq, ads)
        for f, lb in zip(figs, fig_labels):
            p = settings.fig_path(loc, name, lb)
            save_mpl_fig(f, p)


@fig_qoi_valid.command()
def amb_box(
    loc: str = LOC, name: str = "amb_box", flow_qoi: Path = REG_SRC.flow_qoi_dataset
):
    ads = get_ambient_ds_from_cfg(cfg)
    fq = xr.open_dataset(flow_qoi)

    with plt.rc_context({"figure.figsize": BIG_SIZE}):
        for qoi in [dfc.intake, dfc.cv, dfc.dr]:
            p = settings.fig_path(loc, name, qoi)
            fig = plot_ambient_dep_box_per_qoi(qoi, fq, ads)
            save_mpl_fig(fig, p)


@fig_qoi_valid.command()
def no_corr(
    fig_path: Path = settings.fig_path(LOC, "no_corr"),
    df_path: Path = settings.results_path(LOC, "no_corr", ext="csv"),
    flow_qoi: Path = REG_SRC.flow_qoi_dataset,
):
    fq = xr.open_dataset(flow_qoi)
    with plt.rc_context({"figure.figsize": BIG_SIZE}):
        fig, df = plot_no_qoi_corr(fq)
        save_mpl_fig(fig, fig_path)  # pyright: ignore[reportArgumentType]
        make_dir(df_path)
        df.write_csv(df_path)
