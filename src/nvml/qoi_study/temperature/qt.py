from typing import get_args

import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
from utils4plots.colors import DefaultColors

from nvml.constants import DataNames as dn
from nvml.qdim.wind import WindDirectionBinNames
from nvml.qoi_study.q_richardson import CMAP

dc = DefaultColors()
cmap_lst = [
    dc.cool_steel,
    dc.frosted_blue,
    dc.royal_azure,
    dc.indigo,
    dc.dim_grey,
]


def make_qt_plot(qda: xr.DataArray, tda: xr.DataArray):
    # plan median and plan temperature..
    # one per each
    pass

    qda2 = (
        qda.swap_dims({dn.wind_sector: dn.datetime})
        .median(dim=dn.space_ix)
        .median(dim=dn.datetime)
    )
    tda2 = tda.median(dn.datetime)
    ds = xr.merge([qda2, tda2], compat="no_conflicts")
    print(ds)

    f = ds.plot.scatter(
        x=dn.zone_dimless_temp,
        y=dn.zone_dimless_flow,
    )
    return f


def make_qtw_plot(qda: xr.DataArray, tda: xr.DataArray):
    # plan median and plan temperature..
    # one per each
    pass

    qda2 = qda.swap_dims({dn.wind_sector: dn.datetime}).median(dim=dn.space_ix)
    tda2 = tda.median(dn.datetime)
    ds = xr.merge([qda2, tda2], compat="no_conflicts")
    print(ds)

    labels = get_args(WindDirectionBinNames)
    cat = pd.Categorical(ds[dn.wind_sector].values, categories=labels, ordered=True)
    ds2 = (
        ds.assign_coords({dn.wind_sector: (dn.datetime, cat.codes)})
        .swap_dims({dn.datetime: dn.wind_sector})
        .groupby(dn.wind_sector)
        .median()
    )
    print(ds2)

    fig, ax = plt.subplots()
    f = ds2.plot.scatter(
        x=dn.zone_dimless_temp,
        y=dn.zone_dimless_flow,
        hue=dn.wind_sector,
        levels=len(cat.categories),
        cmap=CMAP,
        ax=ax,
        edgecolors="none",
        # alpha=0.5,
        # s=50,
    )
    return fig
