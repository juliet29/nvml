from pathlib import Path
from typing import get_args

import h5py
import joblib
from loguru import logger

from nvml.wl.dataset.gk_dataset import GKDict
from nvml.wl.dataset.labels import LabelSchemeNames
from nvml.wl.kernel.kern_study import KernelNames, kernel_fxs

# try one of three kernels for each label scheme => 12 kernels total
# save with h5py


def save_kernels(gk_graph_path: Path, out_path: Path):

    def make_kernel_matrix(label: LabelSchemeNames, kernel: KernelNames):
        fx = kernel_fxs[kernel]()

        if kernel == "sm" and label == "null":
            graphs = gk_dict[label][0:50]

        elif kernel == "sm":
            graphs = gk_dict[label][0:100]
        else:
            graphs = gk_dict[label]

        matrix = fx.fit_transform(graphs)
        return matrix

    gk_dict: GKDict = joblib.load(gk_graph_path)

    with h5py.File(out_path, "w") as f:
        for kernel_name in kernel_fxs.keys():
            base = f.create_group(kernel_name)

            for label in get_args(LabelSchemeNames):
                logger.info(f"Creating matrix for {kernel_name}, {label}")
                mat = make_kernel_matrix(label, kernel_name)

                base.create_dataset(label, data=mat)
            logger.success(f"Done with kernel {kernel_name}\n\n")
