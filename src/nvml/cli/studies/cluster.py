import xarray as xr
from cyclopts import App

from nvml.cli.studies.helpers import CASE_NAME, get_ambient_ds, get_graph_path
from nvml.cli.studies.paths import ProjectPaths
from nvml.cluster.assemble import make_space_name_by_wind_sector_da
from nvml.constants import DataNames, FileNames

cluster = App("cluster")


@cluster.command()
def fc():
    path = get_graph_path()
    ambient_ds = get_ambient_ds()
    return make_space_name_by_wind_sector_da(CASE_NAME, path, ambient_ds)


@cluster.command()
def fd():
    # open zarr
    ds = xr.open_zarr(ProjectPaths.data.qdim_test / FileNames.zarr)
    da = ds[DataNames.q_dim_median]
    return da
