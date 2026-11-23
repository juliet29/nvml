import matplotlib
import matplotlib.pyplot as plt  # noqa
import seaborn as sns  # noqa
from cyclopts import App
from ptpython.repl import embed

# from nvml.flearn.kernel.prep import GklDatasetMaker
from nvml._study.config.cstore import CONFIGS_DICT
from nvml.io import get_ambient_ds_from_cfg, graph_to_ds, read_graph
from nvml.qoi.defns import (  # noqa
    calc_flow_qois,
    calc_intake,
    calc_interzonal_ratio,
    calc_uniformity_index,
    plan_qdim,
)

matplotlib.use("module://matplotlib-backend-kitty")
q2_app = App("q2")


cfg = CONFIGS_DICT["msd_aug20"]


@q2_app.command()
def ff():
    gp = cfg.get_one_graph_path(0)
    ds = graph_to_ds(gp)
    g = read_graph(gp)
    assert g
    ambient_ds = get_ambient_ds_from_cfg(cfg)
    ds = calc_flow_qois(g, ambient_ds)

    embed(globals(), locals(), vi_mode=True)
