from itertools import product
from pathlib import Path
from typing import NamedTuple, get_args

import h5py
import numpy as np
from loguru import logger
from numpy.typing import NDArray
from sklearn.decomposition import KernelPCA

from nvml.wl.graphs.labels import LabelSchemeNames
from nvml.wl.kernel.interfaces import KernelNames


class PCAId(NamedTuple):
    kernel: KernelNames
    scheme: LabelSchemeNames

    @property
    def name(self):
        cname = f"{self.kernel}/{self.scheme}"
        return cname

    @property
    def path_name(self):
        cname = f"{self.kernel}_{self.scheme}"
        return cname


class PCANames:
    components = "components"
    eigenvalues = "eigenvalues"


class PCAData(NamedTuple):
    input: h5py.Group

    @property
    def components(self) -> NDArray:
        return np.array(self.input[PCANames.components])

    @property
    def eigenvalues(self) -> NDArray:
        return np.array(self.input[PCANames.eigenvalues])


def compute_pca(X: NDArray):
    model = KernelPCA(kernel="precomputed")
    pcs = model.fit_transform(X)
    lambdas = model.eigenvalues_

    return pcs, lambdas


def save_pcas(kernel_path: Path, out_path: Path):
    groups = list(product(get_args(KernelNames), get_args(LabelSchemeNames)))

    kernel_file = h5py.File(kernel_path, "r")

    with h5py.File(out_path, "w") as f:
        for kernel, scheme in groups:
            dataset_name = f"{kernel}/{scheme}"

            logger.info(f"Saving PCA for {dataset_name}")
            X = np.array(kernel_file[dataset_name])

            pcs, eigvalues = compute_pca(X)
            f.create_dataset(
                f"{dataset_name}/{PCANames.components}", data=pcs, compression="gzip"
            )

            f.create_dataset(
                f"{dataset_name}/{PCANames.eigenvalues}",
                data=eigvalues,
                compression="gzip",
            )
