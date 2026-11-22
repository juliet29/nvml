from typing import Literal, get_args

import pandas as pd
import polars as pl
import xarray as xr

from nvml.constants import DataNames as dn

WindDirectionBinNames = Literal["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
WindPolarsEnum = pl.Enum(get_args(WindDirectionBinNames))


class WindDirectionBins:
    edges = [0, 45, 90, 135, 180, 225, 270, 315, 360]
    labels = get_args(WindDirectionBinNames)


def add_wind_sector_coord(ds: xr.Dataset):
    wdb = WindDirectionBins()

    shifted = (ds[dn.wind_dir] + 22.5) % 360  # ← this is what unifies North
    sector = pd.cut(
        shifted, bins=wdb.edges, labels=wdb.labels, include_lowest=True
    )  # .astype(str)  # pyright: ignore[reportAttributeAccessIssue]
    return ds.assign_coords(
        {dn.wind_sector: (dn.datetime, sector)}
    )  # sector coord indexed along the datetime coord


def wind_sector_as_str(ds: xr.Dataset):
    da = ds[dn.wind_sector]
    return ds.assign_coords({dn.wind_sector: (da.dims, da.values.astype(str))})


def wind_sector_as_categorical(ds: xr.Dataset):
    da = ds[dn.wind_sector]
    cat = pd.Categorical(da.values, categories=WindDirectionBins.labels, ordered=True)
    return ds.assign_coords({dn.wind_sector: (da.dims, cat)})
