from pathlib import Path
from typing import NamedTuple

import h5py
import numpy as np
from numpy.typing import NDArray

from nvml.flearn.graphs.labels import LabelSchemeNames
from nvml.flearn.kernel.interfaces import KernelNames


class ClusterNames:
    coeff = "coeff"
    label = "label"
    case_ix = "case_ix"


class ClusterId(NamedTuple):
    kernel: KernelNames
    scheme: LabelSchemeNames
    k: int


class ClusterData(NamedTuple):
    path: Path
    kernel: KernelNames
    scheme: LabelSchemeNames
    k: int

    @property
    def name(self):
        cname = f"{self.kernel}/{self.scheme}/{self.k}"
        return cname

    def get_data(self):
        cluster_file = h5py.File(self.path, "r")
        d = np.array(cluster_file[self.name])
        cluster_file.close()
        return d

    @property
    def labels(self) -> NDArray:
        return self.get_data()[0]

    @property
    def coeffs(self) -> NDArray:
        return self.get_data()[1]

    @classmethod
    def from_id(cls, path: Path, cid: ClusterId):
        return cls(path, *cid)
