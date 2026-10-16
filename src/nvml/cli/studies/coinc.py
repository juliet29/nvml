import numpy as np
import xarray as xr
from cyclopts import App
from icecream import ic
from plan2eplus.ezcase.ez import EZ
from plan2eplus.geometry.directions import WallNormal

from nvml.cli.config.config import MakeConfig
from nvml.cli.config.cstore import CONFIGS_DICT
from nvml.constants import DataNames as dn
from nvml.features.coinc import tabulate_coinc_features
from nvml.features.qoi import calc_qdim_quantity
from nvml.io import get_ambient_data_as_ds, read_graph
from nvml.qdim.angles import wind_angles_to_vector
from nvml.qdim.incident import make_zone_outward_normal_da
from nvml.qdim.intext import get_normals_for_windows_across_zones
from nvml.qdim.wind import add_wind_sector_coord

coinc_app = App("coinc")

cfg = CONFIGS_DICT["msd_aug20"]
CASE_IX = 0
case_name = cfg.get_one_case(CASE_IX)
case_data = cfg.get_one_case_data(CASE_IX)
case_graph_path = cfg.get_one_graph_path(CASE_IX)


def get_ambient_ds(ix: int, cfg: MakeConfig):
    res = get_ambient_data_as_ds(cfg.get_one_case_data(ix).sql)
    res = add_wind_sector_coord(res)
    return res


@coinc_app.command()
def fb():
    angle_da = xr.DataArray(
        data=[0, 1], coords={dn.xy_vector: ["x", "y"]}, dims=dn.xy_vector
    )
    return angle_da

    return wind_angles_to_vector(WallNormal.NORTH.value)
    # ambient_ds = get_ambient_ds(CASE_IX)  # this is coming from helpers
    # n_mask = ambient_ds[dn.wind_sector] == "N"
    # return n_mask
    # first_n_idx = np.argmax()
    #
    # # get a wind direction with wind secto being north..


@coinc_app.command()
def fc():
    ambient_ds = get_ambient_ds(CASE_IX, cfg)  # this is coming from helpers
    G = read_graph(case_graph_path)
    case = EZ(case_data.idf)

    zons = get_normals_for_windows_across_zones(G, case)
    zone_da = make_zone_outward_normal_da(zons)
    angle_da = xr.DataArray(
        data=[0, 1], coords={dn.xy_vector: ["x", "y"]}, dims=dn.xy_vector
    )
    ic("hi", zone_da, angle_da)
    return xr.dot(angle_da, zone_da, dim=dn.xy_vector)

    # wind_da = wind_angle_da_to_vectors(ambient_ds[dn.wind_dir])
    # return ambient_ds
    #


@coinc_app.command()
def fd():
    da = xr.DataArray(
        data=np.array(
            [[0.0, np.nan], [1.0, 0.0], [1.0, np.nan], [0.0, np.nan], [0.0, np.nan]]
        )
    )
    return tabulate_coinc_features(da)


@coinc_app.command()
def fe():
    ambient_ds = get_ambient_ds(CASE_IX, cfg)  # this is coming from helpers
    G = read_graph(case_graph_path)
    return calc_qdim_quantity(G, ambient_ds, "N")
