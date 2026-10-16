from enum import Enum

import numpy as np
import xarray as xr
from plan2eplus.ops.zones.ezobject import WallNormal


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


def get_feature(ct_set: set[CoincidentType]):
    return features.get(
        frozenset(ct_set), lambda: print("Could not find a matching set!")
    )


def calc_plan_coinc(G, case, incident_wind_drn: WallNormal) -> xr.DataArray:
    # TODO ~ fc from studies/coinc -> which goes to modules when done..

    pass


def tabulate_coinc_features(da: xr.DataArray):
    # turn the data into a set
    data = da.data.flatten()
    clean_data = data[~np.isnan(data)]
    number_set = set(clean_data.tolist())

    # map set to CoincidentType enum
    ct_set = {CoincidentType(i) for i in number_set}

    return get_feature(ct_set)
