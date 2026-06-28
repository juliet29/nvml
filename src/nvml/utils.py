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


def xr_to_polars(data: xr.Dataset | xr.DataArray) -> pl.DataFrame:
    if isinstance(data, xr.DataArray):
        assert data.name, "DataArray must be named to become a variable"
        data = data.to_dataset()
    df = pl.from_pandas(data.to_dataframe().reset_index())
    return df.with_columns(pl.col(pl.Datetime).dt.cast_time_unit("us"))
