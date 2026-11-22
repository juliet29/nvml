from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from loguru import logger
from numpy.typing import NDArray

from nvml.flearn.pca.make_pca import PCAData, PCAId


def get_pca_data(pca_path: Path, pca_id: PCAId):
    pca_file = h5py.File(pca_path, "r")
    d = pca_file[pca_id.name]
    assert isinstance(d, h5py.Group)
    pcd = PCAData(d)
    return pcd


def calc_var_ratio(eigenvalues: NDArray):
    evr = eigenvalues / np.sum(eigenvalues)
    evrs = np.sort(evr, descending=True)
    cs = np.cumsum(evrs)
    pass
    return evrs, cs


def plot_scree(pca_path: Path, pca_id: PCAId):
    data = get_pca_data(pca_path, pca_id)

    logger.info(f"Plotting for {pca_id.name}")
    fig, ax = plt.subplots()

    evrs, cs = calc_var_ratio(data.eigenvalues)

    sns.scatterplot(evrs, ax=ax)
    sns.lineplot(cs, ax=ax)  # if possible, different y axis..

    return fig
