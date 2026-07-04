from pathlib import Path

import numpy as np
import xarray as xr

from nvml.cluster.assemble import make_space_name_by_wind_sector_da
from nvml.constants import DataNames as dn
from nvml.constants import FileNames
from nvml.qdim.wind import WindDirectionBins
from nvml.utils import make_dir


def init_zarr(savedir: Path):
    MAX_SPACES_PER_CASE = 100
    N_CASES = 1000

    init_grid = np.zeros((len(WindDirectionBins.labels), MAX_SPACES_PER_CASE, N_CASES))
    dims = [dn.wind_sector, dn.space_ix, dn.case_name]
    coords = {
        dn.wind_sector: WindDirectionBins.labels,
        dn.space_ix: np.arange(
            MAX_SPACES_PER_CASE
        ),  # pre-allocate 100 although no rooms have up to this,
        dn.case_name: [
            str(i) for i in range(N_CASES)
        ],  # number of cases is less than 1k for now
    }

    da = xr.DataArray(
        data=init_grid,
        coords=coords,
        dims=dims,
        name=dn.q_dim_median,
    )

    make_dir(savedir)
    path = savedir / FileNames.zarr  # TODO: change
    da.to_zarr(path, mode="w", compute=False)


def write_to_zarr(
    case_name_idx: int,
    case_name: str,
    zarr_path: Path,
    graph_path: Path,
    ambient_ds_path: Path,
):
    ambient_ds = xr.open_dataset(ambient_ds_path)
    # create the data array to write
    da = make_space_name_by_wind_sector_da(case_name, graph_path, ambient_ds)
    da.expand_dims(dn.case_name).assign_coords({dn.case_name: case_name})

    # case_name_idx = full_da[dn.case_name].values.to_list().index(case_name)
    region_slice = {dn.case_name: slice(case_name_idx, case_name_idx + 1)}

    da.to_zarr(zarr_path, region=region_slice)
