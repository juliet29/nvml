from pathlib import Path

import numpy as np
import xarray as xr
from plyze.flow_graph.interfaces import FlowGraph

from nvml.constants import DataNames as dn
from nvml.io import graph_to_ds
from nvml.qdim.wind import WindDirectionBinNames


def calc_qdim_quantity(
    graph: Path | FlowGraph, ambient_ds: xr.Dataset, wind_sector: WindDirectionBinNames
):

    qoi_ds = graph_to_ds(graph)
    # clean_ds = qoi_ds[~np.isnan(qoi_ds)]

    # TODO: move this check higher, but since q_dim diveides by velocity, we get infinity whenever U = 0 ->
    q_dim = qoi_ds[dn.zone_dimless_flow]

    # check how many nan rows / cols ther are*
    # return q_dim.where(np.isinf(q_dim), drop=True).shape

    # Keep non-nan numbers, otherwise replace with 0
    qdc = q_dim.where(~np.isinf(q_dim), other=0)
    # return q_dim_clean
    # ic(qdc, ambient_ds)

    qdc_wind = (
        qdc.assign_coords(
            {dn.wind_sector: (dn.datetime, ambient_ds[dn.wind_sector].data)}
        )
        .swap_dims({dn.datetime: dn.wind_sector})
        .drop_vars(dn.datetime)
    )

    res = (
        qdc_wind.where(qdc_wind[dn.wind_sector] == wind_sector, drop=True)
        .mean()
        .values  # .tolist()[0]
    )
    return float(res)
