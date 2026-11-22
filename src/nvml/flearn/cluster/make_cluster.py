from itertools import product
from pathlib import Path
from typing import get_args

import h5py
import numpy as np
from loguru import logger
from numpy.typing import NDArray
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_samples

from nvml.settings import settings
from nvml.flearn.graphs.labels import LabelSchemeNames
from nvml.flearn.kernel.interfaces import KernelNames


def cluster(X: NDArray, k: int, random_state: int = settings.random_state):
    model = SpectralClustering(
        n_clusters=k,
        affinity="precomputed",
        random_state=random_state,
        assign_labels="cluster_qr",  # More deterministic labels
    )
    labels: NDArray = model.fit_predict(X)
    return model, labels


def make_cluster_data_with_silhouette_scores(X: NDArray, k: int):
    model, labels = cluster(X, k)
    coeffs: NDArray = silhouette_samples(X, labels)  # pyright: ignore[reportAssignmentType]
    arr = np.vstack([labels, coeffs])
    return arr


def save_clusters(
    kernel_path: Path, out_path: Path, n_clusters: list[int] = [2, 3, 4, 5]
):
    groups = list(
        product(get_args(KernelNames), get_args(LabelSchemeNames), n_clusters)
    )

    kernel_file = h5py.File(kernel_path, "r")

    with h5py.File(out_path, "w") as f:
        for kernel, scheme, k in groups:
            dataset_name = f"{kernel}/{scheme}"

            logger.info(f"Saving clusters for {dataset_name}")
            X = np.array(kernel_file[dataset_name])

            arr = make_cluster_data_with_silhouette_scores(X, k)
            f.create_dataset(f"{dataset_name}/{k}", data=arr, compression="gzip")
