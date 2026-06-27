import xarray as xr

from nvml.qdim.interfaces import ZoneAndOutwardNormals


def make_zone_outward_normal_da(zons: list[ZoneAndOutwardNormals]):
    zone_coord = {"zone": [i.zone_name for i in zons]}
    max_edge_len = max([len(i.edge_and_normals) for i in zons])
    edge_coord = {"edge_num": list(range(max_edge_len))}
    vector_coord = {"vector": ["x", "y"]}
    coords = zone_coord | edge_coord | vector_coord

    dims = ["zone", "edge_num", "vector"]

    data = [i.normal_vectors for i in zons]

    da = xr.DataArray(data=data, coords=coords, dims=dims)
    return da


def make_wind_direction_vector_da(da: xr.DataArray):
    """Vector of standard wind directions (defined as angles in degrees) is expected"""
    return da
    # return wind_angles_to_vector(da)
