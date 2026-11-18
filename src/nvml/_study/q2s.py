import matplotlib
import matplotlib.pyplot as plt  # noqa
import seaborn as sns  # noqa
from cyclopts import App
from ptpython.repl import embed

# from nvml.wl.kernel.prep import GklDatasetMaker
from nvml._study.config.cstore import CONFIGS_DICT
from nvml.io import graph_to_ds

matplotlib.use("module://matplotlib-backend-kitty")
q2_app = App("q2")


cfg = CONFIGS_DICT["msd_aug20"]


@q2_app.command()
def ff():
    gp = cfg.get_one_graph_path(0)
    ds = graph_to_ds(gp)

    embed(globals(), locals(), vi_mode=True)
