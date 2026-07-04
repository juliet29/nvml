from pathlib import Path

from cyclopts import App

from nvml.cluster.multi import init_zarr, write_to_zarr
from nvml.constants import FileNames
from nvml.io import get_ambient_data_as_ds

qdim = App("qdim")


@qdim.command()
def init(savedir: Path, sqlpath: Path):
    init_zarr(savedir)
    ds = get_ambient_data_as_ds(sqlpath)

    ds.to_netcdf(savedir / FileNames.ambient)


@qdim.command()
def create_case_data(
    case_name_idx: int,
    case_name: str,
    zarr_path: Path,
    graph_path: Path,
    ambient_ds_path: Path,
):
    write_to_zarr(case_name_idx, case_name, zarr_path, graph_path, ambient_ds_path)
