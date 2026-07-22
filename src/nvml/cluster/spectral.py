import xarray as xr
from sklearn.cluster import SpectralClustering

from nvml.constants import DataNames as dn

RANDOM_STATE = 1


def make_spectral(da: xr.DataArray, wind_sector: str):
    X = da.sel({dn.wind_sector: wind_sector}).values

    clustering = SpectralClustering(
        n_clusters=2, random_state=RANDOM_STATE, assign_labels="cluster_qr"
    ).fit(X)
    labels = clustering.labels_
    return xr.DataArray(
        labels,
        coords={dn.case_name: da.case_name.values},
    )


def plot_cluster_on_data(labels: xr.DataArray, da: xr.DataArray):
    Yds = (
        da.isel({dn.space_ix: slice(None, 2)})
        .to_dataset(dim=dn.space_ix)
        .assign({dn.label: labels})
    )
    # return Yds
    Yds.plot.scatter(x=0, y=1, hue=dn.label)


def plot_cluster_on_tsne(labels: xr.DataArray, Y_tsne: xr.DataArray):
    Yds = Y_tsne.to_dataset(dim=dn.coord).assign({dn.label: labels})
    return Yds

    # .plot.scatter(x=dn.c1, y=dn.c2)
