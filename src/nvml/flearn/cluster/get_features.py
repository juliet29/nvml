import numpy as np
import xarray as xr

from nvml.flearn.cluster.interfaces import ClusterData, ClusterNames


def create_features(cd: ClusterData):
    cn = ClusterNames

    d = cd.get_data()
    label_data = d[0, :]
    coeff_data = d[0, :]
    ds = xr.Dataset(
        data_vars={cn.label: (cn.label, label_data), cn.coeff: (cn.coeff, coeff_data)},
        coords={cn.case_ix: np.arange(len(label_data))},
    )
    feature = xr.where(ds[cn.coeff] <= 0, -1, ds[cn.label]).data
    return feature
