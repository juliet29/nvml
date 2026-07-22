from cyclopts import App

from nvml.cli.studies.paths import ProjectPaths
from nvml.cluster.spectral import make_spectral, plot_cluster_on_data
from nvml.cluster.tsne import make_tsne, plot_tsne, setup_tsne
from nvml.constants import FileNames
from nvml.utils import make_dir

cluster = App("cluster")
WIND_DIR = "N"


def fc():
    path = ProjectPaths.data.qdim_test / FileNames.zarr
    da = setup_tsne(path)
    return da


@cluster.command()
def fda():
    path = ProjectPaths.data.qdim_test / FileNames.zarr
    da = setup_tsne(path)
    Y = make_tsne(da, WIND_DIR)
    p = ProjectPaths.data.tsne / FileNames.general_nc
    make_dir(p)

    Y.to_netcdf(p)
    return Y


@cluster.command()
def fe():
    p = ProjectPaths.data.tsne / FileNames.general_nc
    plot_tsne(p, ProjectPaths.figs.tsne)


@cluster.command()
def ff():
    path = ProjectPaths.data.qdim_test / FileNames.zarr
    da = setup_tsne(path)
    labels = make_spectral(da, WIND_DIR)
    return labels


@cluster.command()
def fg():
    labels = ff()
    da = fc()
    return plot_cluster_on_data(labels, da)
