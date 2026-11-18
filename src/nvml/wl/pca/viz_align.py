from pathlib import Path
from typing import Literal, get_args

import matplotlib.pyplot as plt
import polars as pl
from loguru import logger
from matplotlib.axes import Axes
from tqdm import tqdm

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


def make_sample_indices_quartile(s: pl.Series, n_samples: int):

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
    return (d["high"], d["mid"], d["low"])


def make_sample_indices_ordered(s: pl.Series, n_samples: int):
    high = s.arg_sort().head(n_samples).to_list()
    low = s.arg_sort().tail(n_samples).reverse().to_list()
    mid = s.abs().arg_sort().head(n_samples).to_list()
    return high, low, mid


def plot_quartile_group(
    cfg: MakeConfig,
    s: pl.Series,
    indices: list[int],
    ax_group: list[Axes],
):

    for ax, ix in zip(ax_group, indices):
        case_name = cfg.get_one_case(ix)
        value = s[ix]
        percentile = (s < value).mean()

        plot_one(cfg, ix, ax, label=f"{case_name} | {int(percentile * 100)}th Perc.")
    pass


def plot_component_graphs(
    cfg: MakeConfig,
    pca_path: Path,
    pca_id: PCAId,
    component_numbers: list[int],
    n_samples: int,
    method: Literal["ordered", "percentile"],
):
    pcd = get_pca_data(pca_path, pca_id)

    ix_fx = (
        make_sample_indices_ordered
        if method == "ordered"
        else make_sample_indices_quartile
    )

    def handle_num(num: int):
        logger.info(f"Beginning figure for component {num}")
        s = pl.Series(pcd.components[:, num])
        index_groups = ix_fx(s, n_samples)
        bin_names = ["high", "mid", "low"]

        fig, ax_groups = plt.subplots(nrows=len(index_groups), ncols=n_samples)
        for name, bin, ax_group in tqdm(zip(bin_names, index_groups, ax_groups)):
            ax_group[0].set_ylabel(f"{name.capitalize()} on Ax {num}")
            plot_quartile_group(cfg, s, bin, ax_group)
        return fig

    return [handle_num(i) for i in tqdm(component_numbers)]
