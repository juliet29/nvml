from pathlib import Path

import numpy as np
import xarray as xr
from sklearn import manifold

from nvml.constants import DataNames as dn


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

    # open zarr, sort array, get first 5 values, and drop the case if ther are not up to 5..
    # really need to look at histogram of the case to say accurantely
    # need apply u_func for this.. => need to write good res for this...

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


def make_tsne(X: np.ndarray):
    tsne = manifold.TSNE(perplexity=10)
    Y = tsne.fit_transform(X)
    return Y


def plot_tsne(Y):
    pass
