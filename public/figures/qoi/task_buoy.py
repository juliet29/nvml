import matplotlib.pyplot as plt
from plyze.qoi.xarray_helpers import CardinalNameLiteral
from utils4plots import set_dynamic_params
from utils4plots.save import save_mpl_fig

from nvml._study.config.cstore import CONFIGS_DICT
from nvml.buoy.data import BuoyData
from nvml.buoy.viz_q import corr_U_t_dim, dists, facet_q_ri, plot_q_dim
from nvml.settings import FigProduct, settings

LOC = "qoi"
cfg = CONFIGS_DICT["msd_aug20"]
BIG_SIZE = (15, 7.5)
set_dynamic_params(settings.figure_profile)


def task_plot_qdim(
    out_path: FigProduct = settings.fig_path(LOC, "qdim"),
):
    CASE_NAME = str(4969)
    ROOM_NAME = "room_2"
    facade: CardinalNameLiteral = "NORTH"
    bd = BuoyData(cfg, CASE_NAME, ROOM_NAME, facade)

    with plt.rc_context({"figure.figsize": BIG_SIZE}):
        fig = plot_q_dim(bd)
        save_mpl_fig(fig, out_path)


def task_plot_q_ri_facets(
    p1: FigProduct = settings.fig_path(LOC, "q_ri_u"),
    p2: FigProduct = settings.fig_path(LOC, "q_ri_tdim"),
):
    CASE_NAME = str(4969)
    ROOM_NAME = "room_2"
    facade: CardinalNameLiteral = "NORTH"
    bd = BuoyData(cfg, CASE_NAME, ROOM_NAME, facade)

    with plt.rc_context({"figure.figsize": BIG_SIZE}):
        f1, f2 = facet_q_ri(bd)
        save_mpl_fig(f1, p1)
        save_mpl_fig(f2, p2)


def task_corr_U_t_dim(
    p1: FigProduct = settings.fig_path(LOC, "U_t_dim"),
):
    CASE_NAME = str(4969)
    ROOM_NAME = "room_2"
    facade: CardinalNameLiteral = "NORTH"
    bd = BuoyData(cfg, CASE_NAME, ROOM_NAME, facade)

    with plt.rc_context({"figure.figsize": BIG_SIZE}):
        f1 = corr_U_t_dim(bd)
        save_mpl_fig(f1, p1)


def task_dists(
    p1: FigProduct = settings.fig_path(LOC, "d_Ri"),
    p2: FigProduct = settings.fig_path(LOC, "d_t_dim"),
    p3: FigProduct = settings.fig_path(LOC, "d_U"),
    p4: FigProduct = settings.fig_path(LOC, "d_q_dim"),
):
    CASE_NAME = str(4969)
    ROOM_NAME = "room_2"
    facade: CardinalNameLiteral = "NORTH"
    bd = BuoyData(cfg, CASE_NAME, ROOM_NAME, facade)

    with plt.rc_context({"figure.figsize": BIG_SIZE}):
        f1, f2, f3, f4 = dists(bd)
        save_mpl_fig(f1, p1)
        save_mpl_fig(f2, p2)
        save_mpl_fig(f3, p3)
        save_mpl_fig(f4, p4)
