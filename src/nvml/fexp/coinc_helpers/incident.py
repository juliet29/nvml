import numpy as np
import xarray as xr

from nvml.constants import DataNames as dn
from nvml.fexp.coinc_helpers.interfaces import ZoneAndOutwardNormals


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
    da = xr.DataArray(data=padded_data, coords=coords, dims=dims)
    return da
