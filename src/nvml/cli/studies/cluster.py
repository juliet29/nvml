import numpy as np
import xarray as xr
from cyclopts import App
from icecream import ic

from nvml.cli.studies.helpers import CASE_NAME, get_ambient_ds, get_graph_path
from nvml.cli.studies.paths import ProjectPaths
from nvml.cluster.setup.single import make_space_name_by_wind_sector_da
from nvml.cluster.tsne import setup_tsne
from nvml.constants import DataNames as dn
from nvml.constants import FileNames

cluster = App("cluster")


@cluster.command()
def fc():
    path = get_graph_path()
    ambient_ds = get_ambient_ds()
    return make_space_name_by_wind_sector_da(CASE_NAME, path, ambient_ds)


@cluster.command()
def uf_tut():
    da = xr.DataArray(
        [[3, 2, 5, 7, np.nan, np.nan], [1, 2, 3, 4, 5, 6]],
        coords={dn.space_ix: np.arange(6), dn.wind_sector: ["N", "NE"]},
        dims=[dn.wind_sector, dn.space_ix],
    )
    # return da.where(np.isfinite(da), drop=True)
    return da.isel({dn.space_ix: slice(None, 5)}).dropna(dim=dn.wind_sector, how="any")


@cluster.command()
def fda():
    path = ProjectPaths.data.qdim_test / FileNames.zarr
    return setup_tsne(path)


@cluster.command()
def fd():
    # open zarr
    ds = xr.open_zarr(ProjectPaths.data.qdim_test / FileNames.zarr).load()
    da = ds[dn.q_dim_median]

    da0 = da.isel({dn.case_name: 0})  # .isel({dn.wind_sector: 0}).squeeze()
    ic(da0)
