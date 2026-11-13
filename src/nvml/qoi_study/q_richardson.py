from typing import get_args

import pandas as pd
import xarray as xr
from matplotlib.colors import LinearSegmentedColormap
from utils4plots import CustomColorObjects
from utils4plots.colors import DefaultColors

from nvml.constants import DataNames as dn
from nvml.qdim.wind import WindDirectionBinNames

dc = DefaultColors()
cmap_lst = [
    dc.cool_steel,
    dc.frosted_blue,
    dc.royal_azure,
    dc.indigo,
    dc.pacific,
    dc.sky_blue,
    # dc.dim_grey,
]

CMAP = LinearSegmentedColormap.from_list("cyclical", cmap_lst)


def q_tdim_wind(qda: xr.DataArray, tda: xr.DataArray):

    n_rooms = 7
    n_cases = 6

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
        alpha=1,
        cmap=CMAP,
    )
    # cbar = f.fig.axes[-1]
    cbar = f.cbar
    print(cbar)
    assert cbar
    cbar.set_ticklabels(labels)
    return f.fig


def q_tdim_plot_rooms(qda: xr.DataArray, tda: xr.DataArray):

    n_rooms = 7
    n_cases = 6

    qda2 = qda.swap_dims({dn.wind_sector: dn.datetime})
    ds = xr.merge([qda2, tda], compat="no_conflicts").isel(
        {dn.case_name: range(n_cases), dn.space_ix: range(n_rooms)}
    )

    labels = get_args(WindDirectionBinNames)
    cat = pd.Categorical(ds[dn.wind_sector].values, categories=labels, ordered=True)
    ds2 = (
        ds.assign_coords({dn.wind_sector: (dn.datetime, cat.codes)}).isel(
            {dn.case_name: range(n_cases)}
        )
        # .resample({dn.datetime: "3h"})
        # .mean()
    )

    f = ds2.plot.scatter(
        x=dn.zone_dimless_temp,
        y=dn.zone_dimless_flow,
        col=dn.case_name,
        hue=dn.space_ix,
        col_wrap=3,
        levels=n_rooms,
        s=50,
        alpha=1,
        cmap=CustomColorObjects.cmap,
    )
    # cbar = f.fig.axes[-1]
    return f.fig


def q_ri_plot(qda: xr.DataArray, tda: xr.DataArray):
    n_rooms = 7
    n_cases = 6

    qda2 = qda.swap_dims({dn.wind_sector: dn.datetime})
    ri = tda / (tda[dn.wind_speed] ** 2)
    ri.name = "Richardson Number"
    ds = xr.merge([qda2, ri], compat="no_conflicts").isel(
        {dn.case_name: range(n_cases), dn.space_ix: range(n_rooms)}
    )

    f = ds.plot.scatter(
        x=ri.name,
        y=dn.zone_dimless_flow,
        col=dn.case_name,
        hue=dn.space_ix,
        col_wrap=3,
        levels=n_rooms,
        s=50,
        alpha=1,
        cmap=CustomColorObjects.cmap.colors,
    )
    return f.fig
