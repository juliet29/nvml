from functools import partial
from typing import Callable, Literal

import numpy as np
from grakel import Graph as grk_graph
from grakel import (  # noqa
    GraphletSampling,
    Kernel,
    ShortestPath,
    SubgraphMatching,
    VertexHistogram,
    WeisfeilerLehman,
)

KFxType = Callable[[list[grk_graph]], np.ndarray]
KernelNames = Literal["wl", "sp", "sm"]


def mkf(grk_class: type[Kernel], **kwargs):
    return grk_class(normalize=True, **kwargs)


kernel_fxs: dict[KernelNames, Callable[[], Kernel]] = {
    "wl": partial(mkf, WeisfeilerLehman),
    "sp": partial(mkf, ShortestPath),
    "sm": partial(mkf, SubgraphMatching, k=4, n_jobs=12, verbose=True),
}
