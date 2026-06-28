from pathlib import Path

import polars as pl
import seaborn.objects as so
import xarray as xr
from matplotlib.figure import Figure


def make_dir(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def save_mpl_fig(fig: Figure, path: Path, dpi: int = 300, **kwargs) -> Path:
    make_dir(path)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", **kwargs)
    return path


def save_seaborn_fig(fig: so.Plot, path: Path, dpi: int = 300, **kwargs) -> Path:
    make_dir(path)
    fig.save(path, dpi=dpi, bbox_inches="tight", **kwargs)
    return path


def dataset_to_polars(ds: xr.Dataset):
    df = pl.from_pandas(ds.to_dataframe().reset_index())
    return df


def dataarray_to_polars(da: xr.DataArray, name: str | None = None):
    if not name:
        assert da.name
    df = pl.from_pandas(da.to_dataframe().reset_index())
    return df
