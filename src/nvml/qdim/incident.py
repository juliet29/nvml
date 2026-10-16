import numpy as np
import xarray as xr
from icecream import ic

from nvml.constants import DataNames as dn
from nvml.qdim.angles import wind_angles_to_vector
from nvml.qdim.interfaces import ZoneAndOutwardNormals


def pad_array(data: list[list[np.ndarray]]):
    lengths = np.array([len(row) for row in data])
    n_max = lengths.max()

    mask = np.arange(n_max) < lengths[:, None]  # shape (5, n_max)

    padded = np.full((len(data), n_max, len(data[0][0])), np.nan)
    padded[mask] = np.concatenate([np.stack(r) for r in data if r])
    return padded


def make_zone_outward_normal_da(zons: list[ZoneAndOutwardNormals]):
    zone_coord = {dn.space_name: [i.zone_name for i in zons]}
    max_edge_len = max([len(i.edge_and_normals) for i in zons])
    edge_coord = {dn.edge_num: list(range(max_edge_len))}
    vector_coord = {dn.xy_vector: ["x", "y"]}
    coords = zone_coord | edge_coord | vector_coord

    dims = [dn.space_name, dn.edge_num, dn.xy_vector]

    data = [
        i.normal_vectors for i in zons
    ]  # TODO: for transparency, could bring the wind_angles_to_vector call up here..

    padded_data = pad_array(data)
    ic(padded_data)
    da = xr.DataArray(data=padded_data, coords=coords, dims=dims)
    return da


def wind_angle_da_to_vectors(wind_direction_degrees: xr.DataArray):
    def fx(x):
        return wind_angles_to_vector(x).T

    return xr.apply_ufunc(
        fx,
        wind_direction_degrees,
        input_core_dims=[[dn.datetime]],
        output_core_dims=[[dn.datetime, dn.xy_vector]],
    ).assign_coords({dn.xy_vector: ["x", "y"]})


def calculate_incidence_factor(zone_da: xr.DataArray, wind_da: xr.DataArray):
    """incidence factor [-1,1] = cos(incidence_angle [0,2pi])"""
    assert zone_da.dims == (dn.space_name, dn.edge_num, dn.xy_vector)
    assert wind_da.dims == (dn.datetime, dn.xy_vector)
    incidence_factor_across_surfaces = zone_da @ wind_da.T
    incidence_factor_per_zone.name = dn.incident_factor

    return incidence_factor_per_zone


def calculate_incidence_factor_for_zone(zone_da: xr.DataArray, wind_da: xr.DataArray):
    """incidence factor [-1,1] = cos(incidence_angle [0,2pi])"""
    assert zone_da.dims == (dn.space_name, dn.edge_num, dn.xy_vector)
    assert wind_da.dims == (dn.datetime, dn.xy_vector)
    incidence_factor_across_surfaces = zone_da @ wind_da.T
