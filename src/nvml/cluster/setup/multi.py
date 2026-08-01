from pathlib import Path

import numpy as np
import xarray as xr
from plyze.flow_graph.interfaces import FlowGraph

from nvml.cluster.setup.single import make_space_name_by_wind_sector_da
from nvml.constants import DataNames as dn
from nvml.constants import FileNames
from nvml.io import get_ambient_data_as_ds
from nvml.qdim.wind import WindDirectionBins, add_wind_sector_coord
from nvml.utils import make_dir

MAX_SPACES_PER_CASE = 100  # pre-allocate; no case has this many rooms


def init_zarr(savedir: Path, case_names: list[str]):
    grid = np.zeros(
        (len(WindDirectionBins.labels), MAX_SPACES_PER_CASE, len(case_names))
    )
    dims = [dn.wind_sector, dn.space_ix, dn.case_name]
    coords = {
        dn.wind_sector: list(WindDirectionBins.labels),
        dn.space_ix: np.arange(MAX_SPACES_PER_CASE),
        dn.case_name: case_names,
    }

    da = xr.DataArray(data=grid, coords=coords, dims=dims, name=dn.q_dim_median)

    make_dir(savedir)
    path = savedir / FileNames.zarr
    da.chunk({dn.case_name: 1}).to_zarr(path, mode="w", compute=False)


def write_to_zarr(
    case_name: str,
    zarr_path: Path,
    graph: Path | FlowGraph,
    ambient_ds_or_sql_path: Path | xr.Dataset,
):
    ambient_ds = get_ambient_data_as_ds(ambient_ds_or_sql_path).pipe(
        add_wind_sector_coord
    )  # TODO: should pass this dataset directly..
    da = make_space_name_by_wind_sector_da(case_name, graph, ambient_ds)
    da = (
        da.reindex(
            {
                dn.wind_sector: WindDirectionBins.labels,
                dn.space_ix: np.arange(MAX_SPACES_PER_CASE),
            },
            fill_value=np.nan,
        )
        .expand_dims(dn.case_name)
        .assign_coords({dn.case_name: [case_name]})
    )
    da.to_zarr(zarr_path, region="auto")
