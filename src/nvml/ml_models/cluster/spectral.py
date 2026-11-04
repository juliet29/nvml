import xarray as xr
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score

from nvml.constants import DataNames as dn

RANDOM_STATE = 1


def make_spectral(
    da: xr.DataArray,
    wind_sector: str,
    n_clusters: int = 2,
    random_state: int = RANDOM_STATE,
):
    X = da.sel({dn.wind_sector: wind_sector}).values

    clustering = SpectralClustering(
        n_clusters=n_clusters, random_state=random_state, assign_labels="cluster_qr"
    ).fit(X)
    labels = clustering.labels_
    return xr.DataArray(
        labels,
        coords={dn.case_name: da.case_name.values},
    )


def silhouette_sweep(
    da: xr.DataArray,
    wind_sector: str,
    k_values: list[int],
    random_state: int = RANDOM_STATE,
):
    X = da.sel({dn.wind_sector: wind_sector}).values

    scores = {}
    for k in k_values:
        labels = SpectralClustering(
            n_clusters=k, random_state=random_state, assign_labels="cluster_qr"
        ).fit_predict(X)
        scores[k] = float(silhouette_score(X, labels))
    return scores


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
