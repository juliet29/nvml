from typing import get_args

import pandas as pd
import xarray as xr

from nvml.constants import DataNames as dn
from nvml.qdim.wind import WindDirectionBinNames
from nvml.qoi_study.q_richardson import CMAP

n_cases = 6
n_rooms = 7


def make_scatter_zone_plot(qda: xr.DataArray, tda: xr.DataArray):
    qda2 = qda.swap_dims({dn.wind_sector: dn.datetime})
    ds = xr.merge([qda2, tda], compat="no_conflicts").isel(
        {dn.case_name: range(n_cases), dn.space_ix: range(n_rooms)}
    )

    labels = get_args(WindDirectionBinNames)
    cat = pd.Categorical(ds[dn.wind_sector].values, categories=labels, ordered=True)
    ds2 = ds.assign_coords({dn.wind_sector: (dn.datetime, cat.codes)}).isel(
        {dn.case_name: range(n_cases)}
    )

    f = ds2.plot.scatter(
        x=dn.zone_dimless_temp,
        y=dn.zone_dimless_flow,
        col=dn.case_name,
        hue=dn.wind_sector,
        col_wrap=3,
        levels=len(cat.categories) + 1,
        s=50,
        cmap=CMAP,
        linewidth=0.01,
        edgecolors="black",
    )
    # cbar = f.fig.axes[-1]
    cbar = f.cbar
    print(cbar)
    assert cbar
    cbar.set_ticklabels(labels)
    return f.fig


def add_wind_sector_codes(da: xr.DataArray):
    labels = get_args(WindDirectionBinNames)
    cat = pd.Categorical(da[dn.wind_sector].values, categories=labels, ordered=True)
    ds2 = da.assign_coords({dn.wind_code: (dn.wind_sector, cat.codes)})
    return ds2


def make_3d_zone_plot(qda: xr.DataArray, tda: xr.DataArray):
    qda2 = qda.swap_dims({dn.wind_sector: dn.datetime})
    ds = xr.merge([qda2, tda], compat="no_conflicts").isel(
        {dn.case_name: range(n_cases), dn.space_ix: range(n_rooms)}
    )

    labels = get_args(WindDirectionBinNames)
    cat = pd.Categorical(ds[dn.wind_sector].values, categories=labels, ordered=True)
    ds2 = ds.assign_coords({dn.wind_sector: (dn.datetime, cat.codes)}).isel(
        {dn.case_name: range(n_cases)}
    )

    f = ds2.plot.scatter(
        x=dn.zone_dimless_temp,
        y=dn.zone_dimless_flow,
        col=dn.case_name,
        hue=dn.wind_sector,
        col_wrap=3,
        levels=len(cat.categories) + 1,
        s=50,
        cmap=CMAP,
        linewidth=0.01,
        edgecolors="black",
    )
    # cbar = f.fig.axes[-1]
    cbar = f.cbar
    print(cbar)
    assert cbar
    cbar.set_ticklabels(labels)
    return f.fig

    pass
