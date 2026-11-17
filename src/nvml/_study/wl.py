import matplotlib
from cyclopts import App
from grakel import (  # noqa
    ShortestPath,
    SubgraphMatching,
    VertexHistogram,
    WeisfeilerLehman,
)

from nvml import DataPaths

# from nvml.wl.kernel.prep import GklDatasetMaker
from nvml._study.config.cstore import CONFIGS_DICT
from nvml.wl.graphs.gk_dataset import ManyLabelsGklDatasetMaker

matplotlib.use("module://matplotlib-backend-kitty")
wl3_app = App("wl3")

cfg = CONFIGS_DICT["msd_aug20"]


@wl3_app.command()
def fe1():
    dir_path = DataPaths.kern.v2.gk_graphs_joblib(test=True)

    rng = (0, 10)
    n_jobs = 5
    gkdm = ManyLabelsGklDatasetMaker(
        cfg, dir_path, true_run=False, save=False, test_rng=rng, njobs=n_jobs
    )
    gkdm.run()
    return gkdm
    # graph_list = gkdm.mod_df["int_ext"]
    # # kern = SubgraphMatching(normalize=True).fit_transform(graph_list)
    # kern = SubgraphMatching(normalize=True).fit_transform(graph_list)
    # ic(kern.shape)
