from pathlib import Path

from cyclopts import App

from nvml.cluster.multi import init_zarr, write_to_zarr
from nvml.constants import FileNames
from nvml.io import get_ambient_data_as_ds
from nvml.qdim.wind import add_wind_sector_coord, wind_sector_as_str

qdim = App("qdim")


@qdim.command()
def init(savedir: Path, sqlpath: Path, case_names: list[str]):
    init_zarr(savedir, case_names)
    ds = (
        get_ambient_data_as_ds(sqlpath)
        .pipe(add_wind_sector_coord)
        .pipe(wind_sector_as_str)
    )

    ds.to_netcdf(savedir / FileNames.ambient)


@qdim.command()
def create(
    case_name: str,
    zarr_path: Path,
    graph_path: Path,
    ambient_ds_path: Path,
):
    write_to_zarr(case_name, zarr_path, graph_path, ambient_ds_path)
