from itertools import product
from pathlib import Path

from nvml.regression.models.collate import prepare_pca_df
from nvml.wl.pca.pca_data import PCAId


def make_pca_dfs(path: Path, n: int):
    pca_pairs = list(product(["wl", "sp"], ["int_ext", "area_ratio"]))
    pca_dfs = {pair: prepare_pca_df(path, PCAId(*pair), n) for pair in pca_pairs}
    return pca_dfs
