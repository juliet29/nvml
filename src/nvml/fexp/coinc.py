from enum import Enum
from typing import NamedTuple, get_args

import polars as pl
import xarray as xr
from plan2eplus.ezcase.ez import EZ
from plan2eplus.ops.zones.ezobject import WallNormal
from plyze.flow_graph.interfaces import FlowGraph
from plyze.qoi.xarray_helpers import CardinalNameLiteral

from nvml.constants import DataNames as dn
from nvml.fexp.coinc_helpers.angles import wind_angles_to_vector
from nvml.fexp.coinc_helpers.incident import make_zone_outward_normal_da
from nvml.fexp.coinc_helpers.intext import get_normals_for_windows_across_zones


class CoincidentType(Enum):
    windward = -1
    leeward = 1
    side = 0


class CoincidentFeature(Enum):
    ww = 1
    lw = 2
    sd = 3
    ww_lw = 4
    ww_sd = 5
    lw_sd = 6
    ww_lw_sd = 7


CT = CoincidentType
CF = CoincidentFeature
features = {
    frozenset({CT.windward}): CF.ww,
    frozenset({CT.leeward}): CF.lw,
    frozenset({CT.side}): CF.sd,
    frozenset({CT.windward, CT.leeward}): CF.ww_lw,
    frozenset({CT.windward, CT.side}): CF.ww_sd,
    frozenset({CT.leeward, CT.side}): CF.lw_sd,
    frozenset({CT.windward, CT.leeward, CT.side}): CF.ww_lw_sd,
}


COINCIDENCE_POLARS_ENUM = pl.Enum(
    categories=["ww", "lw", "sd", "ww_lw", "ww_sd", "lw_sd", "ww_lw_sd"]
)


class CoincidenceTuple(NamedTuple):
    windward: bool
    leeward: bool
    side: bool


def get_feature(ct_set: set[CoincidentType]):
    res = features.get(frozenset(ct_set))
    assert res, f"Could not find reature for this set: {ct_set}"
    return res


def calc_plan_coinc(
    G: FlowGraph, case: EZ, incident_wind_drn: CardinalNameLiteral
) -> xr.DataArray:
    zons = get_normals_for_windows_across_zones(G, case)
    zone_da = make_zone_outward_normal_da(zons)

    wind_value = WallNormal[incident_wind_drn].value
    angle_drn = wind_angles_to_vector(wind_value)

    angle_da = xr.DataArray(
        data=angle_drn, coords={dn.xy_vector: ["x", "y"]}, dims=dn.xy_vector
    )

    da = xr.dot(angle_da, zone_da, dim=dn.xy_vector)
    return da


def tabulate_coinc_features(da: xr.DataArray):
    data = da.data.flatten().tolist()
    number_set = set(data)

    ct = CoincidentType
    windward = ct.windward.value in number_set
    leeward = ct.leeward.value in number_set
    side = ct.side.value in number_set
    return CoincidenceTuple(windward, leeward, side)


def get_coinc_for_all_normal_drns(G: FlowGraph, case: EZ):

    def mk(drn):
        coinc_da = calc_plan_coinc(G, case, drn)
        return tabulate_coinc_features(coinc_da)

    return {drn: mk(drn) for drn in get_args(CardinalNameLiteral)}
