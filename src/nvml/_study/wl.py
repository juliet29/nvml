import matplotlib
from cyclopts import App
from grakel import (  # noqa
    ShortestPath,
    SubgraphMatching,
    VertexHistogram,
    WeisfeilerLehman,
)
from ptpython.repl import embed

from nvml import DataPaths

# from nvml.wl.kernel.prep import GklDatasetMaker
from nvml._study.config.cstore import CONFIGS_DICT
from nvml.wl.graphs.gk_dataset import ManyLabelsGklDatasetMaker
from nvml.wl.pca.pca_data import PCAId
from nvml.wl.pca.viz_scree import get_pca_data

matplotlib.use("module://matplotlib-backend-kitty")
wl_app = App("wl")

cfg = CONFIGS_DICT["msd_aug20"]


@wl_app.command()
def fe1():
    dir_path = DataPaths.kern.v2.gk_graphs_joblib(test=True)

    rng = (0, 10)
    n_jobs = 5
    gkdm = ManyLabelsGklDatasetMaker(
        cfg, dir_path, true_run=False, save=False, test_rng=rng, njobs=n_jobs
    )
    gkdm.run()
    return gkdm


@wl_app.command()
def ff():
    p = DataPaths.kern.v2.pca_h5
    pair = PCAId("wl", "int_ext")
    pcd = get_pca_data(p, pair)
    embed(globals(), locals(), vi_mode=True)
