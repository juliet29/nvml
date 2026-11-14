import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from cyclopts import App
from grakel import (  # noqa
    ShortestPath,
    SubgraphMatching,
    VertexHistogram,
    WeisfeilerLehman,
)
from icecream import ic
from plyze.qoi.xarray_helpers import xr
from utils4plans.io.base import make_dir

from nvml._study._root.study_paths import StudyPaths
from nvml._study.config.cstore import CONFIGS_DICT

# from nvml.wl.kernel.prep import GklDatasetMaker
from nvml.constants import FileNames
from nvml.io import read_graph
from nvml.wl.kernel.kern_study import try_clustering_and_plot
from nvml.wl.kernel.prep_many import ManyLabelsGklDatasetMaker
from nvml.wl.kernel.viz import plot_kernel_dist, plot_kernel_heat
from nvml.wl.sil import cluster

matplotlib.use("module://matplotlib-backend-kitty")
wl2_app = App("wl2")


cfg = CONFIGS_DICT["msd_aug20"]


def get_kern():
    return np.load(StudyPaths.data.kern.nov13 / FileNames.sm_kernel_npy)


@wl2_app.command()
def fe1():
    dir_path = StudyPaths.data.kern.nov13
    fn = "fake.csv"

    rng = (0, 800)
    n_jobs = 10
    gkdm = ManyLabelsGklDatasetMaker(
        cfg, dir_path, fn, true_run=True, test_rng=rng, njobs=n_jobs
    )
    gkdm.run()
    graph_list = gkdm.mod_df["int_ext"]
    # kern = SubgraphMatching(normalize=True).fit_transform(graph_list)
    kern = SubgraphMatching(normalize=True).fit_transform(graph_list)
    ic(kern.shape)

    kern_save_path = dir_path / FileNames.sm_kernel_npy
    make_dir(kern_save_path)
    print(f"Saving to {kern_save_path}")
    np.save(kern_save_path, kern, False)
    print("Saved!")


@wl2_app.command()
def fe2():
    kern = get_kern()

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, layout="constrained")
    plot_kernel_dist(kern, ax1)  # pyright: ignore[reportArgumentType]
    plot_kernel_heat(kern, ax2)  # pyright: ignore[reportArgumentType]

    top_k = try_clustering_and_plot(kern, ax3, k_to_try=[2, 3, 4, 5, 6, 7])
    print(top_k)

    plt.show()


@wl2_app.command()
def fe25():
    gs = [read_graph(cfg.get_one_graph_path(i)) for i in range(300)]
    n_nodes = [len(i.zone_names) for i in gs if i]
    df = pl.DataFrame(n_nodes)
    print(df.describe())


@wl2_app.command()
def fe3():
    kern = get_kern()
    ic(kern.shape)
    model, labels = cluster(kern, 2)

    qoi = xr.open_dataarray(StudyPaths.data.reg.aug20v2 / FileNames.qoi_dataArray)
    features = pl.read_csv(StudyPaths.data.reg.aug20v2 / FileNames.regression_data)
    return labels, qoi, features, kern
