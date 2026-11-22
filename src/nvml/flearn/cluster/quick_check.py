from pathlib import Path
from typing import NamedTuple

import h5py
import numpy as np
from numpy.typing import NDArray


class ClusterData(NamedTuple):
    input: h5py.Dataset

    @property
    def data(self):
        return np.array(self.input)

    @property
    def labels(self) -> NDArray:
        return self.data[0]

    @property
    def coeffs(self) -> NDArray:
        return self.data[1]


# TO UTIlS!
def make_flat(f: h5py.File):
    data_dict = {}
    f.visititems(
        lambda name, obj: (
            data_dict.update({name: np.array(obj)})
            if isinstance(obj, h5py.Dataset)
            else None
        )
    )
    return data_dict


def see_sil_scores(cluster_path: Path):
    cluster_file = h5py.File(cluster_path, "r")

    data_dict = make_flat(cluster_file)

    for k, v in data_dict.items():
        coeffs = v[1]
        print(f"{k}: {np.mean(coeffs)}")
    return data_dict
