from pathlib import Path

import xarray as xr
from plyze.flow_graph.interfaces import FlowGraph

from nvml.constants import DataNames as dn
from nvml.io import graph_to_ds
from nvml.qdim.wind import WindDirectionBinNames


def calc_qdim_quantity(
    graph: Path | FlowGraph, ambient_ds: xr.Dataset, wind_sector: WindDirectionBinNames
):

    qoi_ds = graph_to_ds(graph)

    q_dim = qoi_ds[dn.zone_dimless_flow]

    qdc = q_dim

    qdc_wind = (
        qdc.assign_coords(
            {
                dn.wind_sector: (dn.datetime, ambient_ds[dn.wind_sector].data),
                dn.wind_speed: (dn.datetime, ambient_ds[dn.wind_speed].data),
            }
        )
        .swap_dims({dn.datetime: dn.wind_sector})
        .drop_vars(dn.datetime)
    )

    res = qdc_wind.where(qdc_wind[dn.wind_speed] != 0, drop=True)

    return res.where(res[dn.wind_sector] == wind_sector, drop=True)
    return res
    return float(res)
