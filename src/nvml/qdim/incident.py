import xarray as xr

from nvml.constants import DataNames as dn
from nvml.qdim.angles import wind_angles_to_vector
from nvml.qdim.interfaces import ZoneAndOutwardNormals


def make_zone_outward_normal_da(zons: list[ZoneAndOutwardNormals]):
    zone_coord = {dn.space_name: [i.zone_name for i in zons]}
    max_edge_len = max([len(i.edge_and_normals) for i in zons])
    edge_coord = {dn.edge_num: list(range(max_edge_len))}
    vector_coord = {dn.xy_vector: ["x", "y"]}
    coords = zone_coord | edge_coord | vector_coord

    dims = [dn.space_name, dn.edge_num, dn.xy_vector]

    data = [i.normal_vectors for i in zons]

    da = xr.DataArray(data=data, coords=coords, dims=dims)
    return da


def wind_angle_da_to_vectors(angles: xr.DataArray):
    fx = lambda x: wind_angles_to_vector(x).T
    return xr.apply_ufunc(
        fx,
        angles,
        input_core_dims=[[dn.datetime]],
        output_core_dims=[[dn.datetime, dn.xy_vector]],
    )
