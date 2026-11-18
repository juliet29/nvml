from pathlib import Path
from typing import Literal, get_args

import matplotlib.pyplot as plt
import polars as pl
from matplotlib.axes import Axes

from nvml._study.config.config import MakeConfig
from nvml.settings import settings
from nvml.wl.pca.pca_data import PCAId
from nvml.wl.pca.viz_scree import get_pca_data
from nvml.wl.viz import plot_one

# class SampleGroups:
#     low = "low"
#     mid = "mid"
#     high = "high"
SampleGroups = Literal["low", "a1", "mid", "a2", "high"]


def make_sample_indices(s: pl.Series, n_samples: int):

    quantiles = [0.125, 0.4375, 0.5625, 0.875]
    labels = get_args(SampleGroups)
    bins = s.qcut(quantiles=quantiles, labels=labels)

    d: dict[SampleGroups, list[int]] = {
        label: (bins == label)
        .arg_true()
        .sample(
            n=min(n_samples, int((bins == label).sum())), seed=settings.random_state
        )
        .to_list()
        for label in labels
    }
    return (d["low"], d["mid"], d["high"])


def plot_quartile_group(
    cfg: MakeConfig, s: pl.Series, indices: list[int], ax_group: list[Axes]
):

    for ax, ix in zip(ax_group, indices):
        case_name = cfg.get_one_case(ix)
        value = s[ix]
        percentile = (s < value).mean()

        plot_one(cfg, ix, ax, label=f"{case_name} | {int(percentile * 100)}")
    pass


def plot_component_graphs(
    cfg: MakeConfig,
    pca_path: Path,
    pca_id: PCAId,
    component_num: list[int],
    n_samples: int = 3,
):
    pcd = get_pca_data(pca_path, pca_id)

    def handle_num(num: int):
        s = pl.Series(pcd.components[:, num])
        bins = make_sample_indices(s, n_samples)
        bin_names = ["low", "mid", "high"]

        fig, ax_groups = plt.subplots(nrows=len(bins), ncols=n_samples)
        for name, bin in zip(bin_names, bins):
            plot_quartile_group(cfg, s, bin)

    pass
