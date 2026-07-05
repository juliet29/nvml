from pathlib import Path

import numpy as np
import xarray as xr
from sklearn import manifold

from nvml.constants import DataNames as dn
from nvml.constants import FileNames
from nvml.utils import save_mpl_fig


def sort_da_values(da: xr.DataArray, dim: str):
    def fx(
        arr: list[np.ndarray],
    ):
        return np.sort(arr, descending=True, axis=-1)

    return xr.apply_ufunc(
        fx,
        da,
        input_core_dims=[[dim]],
        output_core_dims=[[dim]],
        dask="parallelized",
    )


def setup_tsne(path: Path):
    ds = xr.open_zarr(path)
    da = ds[dn.q_dim_median].load()

    da5 = (
        da.transpose(dn.wind_sector, dn.case_name, dn.space_ix)
        .dropna(dim=dn.wind_sector, how="all")
        .pipe(sort_da_values, dim=dn.space_ix)
        .isel({dn.space_ix: slice(None, 5)})
        .dropna(dim=dn.case_name, how="any")
    )
    return da5


def make_tsne(da: xr.DataArray, wind_sector: str):
    X = da.sel({dn.wind_sector: wind_sector}).values
    tsne = manifold.TSNE(perplexity=15)
    Y = tsne.fit_transform(X)
    return xr.DataArray(
        Y,
        coords={dn.case_name: da.case_name.values, dn.coord: [dn.c1, dn.c2]},
        dims=[dn.case_name, dn.coord],
    )


def plot_tsne(tsne_path: Path, savedir: Path):
    Y = xr.open_dataarray(tsne_path)
    fig = Y.to_dataset(dim=dn.coord).plot.scatter(x=dn.c1, y=dn.c2)
    save_mpl_fig(fig, savedir / FileNames.general_fig)
