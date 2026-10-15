import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from loguru import logger
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from nvml.constants import DataNames as dn
from nvml.style import (
    BLUSHED_BRICK,
    BROWN,
    CORAL,
    DUSTY_TAUPE,
    INK,
    SEQUENTIAL_CMAP,
    class_color,
)

WIND_SECTOR_COLOR = {
    "N": CORAL,
    "E": BLUSHED_BRICK,
    "S": DUSTY_TAUPE,
    "W": BROWN,
}
from nvml.utils import save_mpl_fig


@dataclass
class EpochMetrics:
    epoch: int
    train_loss: float
    train_accuracy: float
    val_balanced_accuracy: float | None = None


@dataclass
class TrainingWatcher:
    log_every: int = 10
    save_path: Path | None = None
    save_every: int = 5
    show_live: bool = False
    make_live_figure: Callable[[], Figure] | None = None
    history: list[EpochMetrics] = field(default_factory=list)

    def record(self, metrics: EpochMetrics):
        self.history.append(metrics)
        if metrics.epoch % self.log_every == 0:
            self.report_latest()
        if metrics.epoch % self.save_every == 0:
            if self.save_path:
                self.save_live_figure(self.save_path)
            if self.show_live:
                self.show_live_figure()

    def current_figure(self) -> Figure:
        if self.make_live_figure:
            return self.make_live_figure()
        return VisualizeTraining(self).make_plot()

    def save_live_figure(self, save_path: Path):
        fig = self.current_figure()
        save_mpl_fig(fig, save_path)
        plt.close(fig)

    def show_live_figure(self):
        fig = self.current_figure()
        plt.show()
        plt.close(fig)

    def report_latest(self):
        m = self.history[-1]
        val = (
            ""
            if m.val_balanced_accuracy is None
            else f" val_balanced_acc: {m.val_balanced_accuracy:.3f}"
        )
        logger.info(
            f"Epoch [{m.epoch}] loss: {m.train_loss:.4f} "
            f"train_acc: {m.train_accuracy:.3f}{val}"
        )

    def report_summary(self):
        final = self.history[-1]
        logger.info(
            f"Final epoch {final.epoch}: loss {final.train_loss:.4f} "
            f"train_acc {final.train_accuracy:.3f}"
        )
        validated = [m for m in self.history if m.val_balanced_accuracy is not None]
        if validated:
            best = max(validated, key=lambda m: m.val_balanced_accuracy or 0.0)
            logger.info(
                f"Best val_balanced_acc {best.val_balanced_accuracy:.3f} "
                f"at epoch {best.epoch}"
            )


class VisualizeTraining:
    def __init__(self, watcher: TrainingWatcher):
        self.watcher = watcher

    def plot_loss(self, ax: Axes):
        history = self.watcher.history
        ax.plot(
            [m.epoch for m in history],
            [m.train_loss for m in history],
            label="train loss",
        )
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        ax.set_title("convergence")
        ax.legend()

    def plot_accuracy(self, ax: Axes):
        history = self.watcher.history
        ax.plot(
            [m.epoch for m in history],
            [m.train_accuracy for m in history],
            label="train accuracy",
        )
        validated = [m for m in history if m.val_balanced_accuracy is not None]
        if validated:
            ax.plot(
                [m.epoch for m in validated],
                [m.val_balanced_accuracy for m in validated],
                label="val balanced acc",
            )
        ax.set_xlabel("epoch")
        ax.set_ylabel("accuracy")
        ax.set_ylim(0, 1)
        ax.set_title("accuracy")
        ax.legend()

    def make_plot(self) -> Figure:
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 5))
        self.plot_loss(axes[0])
        self.plot_accuracy(axes[1])
        fig.tight_layout()
        return fig


class VisualizeCrossValidationConvergence:
    def __init__(self, fold_watchers: list[TrainingWatcher]):
        self.fold_watchers = fold_watchers

    def make_plot(self) -> Figure:
        n_folds = len(self.fold_watchers)
        fig, axes = plt.subplots(
            nrows=n_folds, ncols=2, figsize=(14, 4 * n_folds), squeeze=False
        )
        for fold, watcher in enumerate(self.fold_watchers):
            training = VisualizeTraining(watcher)
            training.plot_loss(axes[fold][0])
            training.plot_accuracy(axes[fold][1])
            axes[fold][0].set_title(f"fold {fold + 1} — convergence")
            axes[fold][1].set_title(f"fold {fold + 1} — accuracy")
        fig.tight_layout()
        return fig


class VisualizeConfusion:
    def __init__(self, confusion: list[list[int]]):
        self.confusion = confusion

    def plot(self, ax: Axes):
        matrix = np.array(self.confusion)
        n_classes = matrix.shape[0]
        ax.imshow(matrix, cmap=SEQUENTIAL_CMAP)
        ax.set_xticks(range(n_classes))
        ax.set_yticks(range(n_classes))
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        ax.set_title("confusion")
        for true_class in range(n_classes):
            for predicted_class in range(n_classes):
                ax.text(
                    predicted_class,
                    true_class,
                    str(matrix[true_class, predicted_class]),
                    ha="center",
                    va="center",
                )

    def make_plot(self) -> Figure:
        fig, ax = plt.subplots(figsize=(6, 5))
        self.plot(ax)
        fig.tight_layout()
        return fig


class VisualizeSilhouette:
    def __init__(self, scores: dict[int, float]):
        self.scores = scores

    def plot(self, ax: Axes):
        k_values = sorted(self.scores)
        best_k = max(self.scores, key=self.scores.get)
        ax.plot(k_values, [self.scores[k] for k in k_values], marker="o")
        ax.axvline(best_k, linestyle="--", color=INK, label=f"best k={best_k}")
        ax.set_xlabel("n_clusters")
        ax.set_ylabel("silhouette")
        ax.set_xticks(k_values)
        ax.set_title("spectral silhouette vs n_clusters")
        ax.legend()

    def make_plot(self) -> Figure:
        fig, ax = plt.subplots(figsize=(7, 5))
        self.plot(ax)
        fig.tight_layout()
        return fig


class VisualizeClusters:
    def __init__(self, labels, qdim_medians, qdim_slopes, ranked_qdim):
        self.labels = np.asarray(labels)
        self.qdim_medians = np.asarray(qdim_medians)
        self.qdim_slopes = np.asarray(qdim_slopes)
        self.ranked_qdim = np.asarray(ranked_qdim)

    def distribution_by_class(self, values, ylabel: str, title: str) -> Figure:
        classes = sorted(np.unique(self.labels).tolist())
        per_class = [values[self.labels == c] for c in classes]

        fig, ax = plt.subplots(figsize=(7, 5))
        parts = ax.violinplot(per_class, showmedians=True)

        def color_body(body, class_label: int):
            body.set_facecolor(class_color(class_label))
            body.set_edgecolor(INK)
            body.set_alpha(0.7)

        for body, class_label in zip(parts["bodies"], classes):
            color_body(body, int(class_label))
        ax.set_xticks(range(1, len(classes) + 1))
        ax.set_xticklabels([f"Class {c}" for c in classes])
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        fig.tight_layout()
        return fig

    def make_qdim_distribution_plot(self) -> Figure:
        return self.distribution_by_class(
            self.qdim_medians,
            "q_dim (5-zone median)",
            "q_dim distribution by assigned cluster",
        )

    def make_qdim_slope_distribution_plot(self) -> Figure:
        return self.distribution_by_class(
            self.qdim_slopes,
            "q_dim slope (rank-1 − rank-5)",
            "q_dim slope distribution by assigned cluster",
        )

    def make_level_evenness_plot(self) -> Figure:
        fig, ax = plt.subplots(figsize=(7, 6))
        for label in sorted(np.unique(self.labels).tolist()):
            mask = self.labels == label
            color = class_color(int(label))
            ax.scatter(
                self.qdim_medians[mask],
                self.qdim_slopes[mask],
                s=55,
                alpha=0.5,
                color=color,
                label=f"Class {label}",
            )
            ax.scatter(
                self.qdim_medians[mask].mean(),
                self.qdim_slopes[mask].mean(),
                s=420,
                marker="X",
                color=color,
                edgecolor=INK,
                linewidth=1.5,
                zorder=5,
            )
        ax.set_xlabel("level — q_dim (5-zone median)")
        ax.set_ylabel("unevenness — q_dim slope (rank-1 − rank-5)")
        ax.set_title("cluster archetypes: level vs evenness")
        ax.legend()
        fig.tight_layout()
        return fig

    def make_rank_profile_plot(self) -> Figure:
        n_ranks = self.ranked_qdim.shape[1]
        ranks = np.arange(1, n_ranks + 1)

        fig, ax = plt.subplots(figsize=(7, 5))
        for label in sorted(np.unique(self.labels).tolist()):
            profiles = self.ranked_qdim[self.labels == label]
            median_profile = np.median(profiles, axis=0)
            low = np.percentile(profiles, 25, axis=0)
            high = np.percentile(profiles, 75, axis=0)
            color = class_color(int(label))
            ax.plot(
                ranks,
                median_profile,
                marker="o",
                markersize=11,
                color=color,
                label=f"Class {label}",
            )
            ax.fill_between(ranks, low, high, alpha=0.2, color=color)
        ax.set_xlabel("zone rank (1 = best-ventilated)")
        ax.set_ylabel("q_dim")
        ax.set_xticks(ranks)
        ax.set_title("q_dim rank profile by assigned cluster (median, IQR band)")
        ax.legend()
        fig.tight_layout()
        return fig


class VisualizeQDimByWind:
    def __init__(
        self,
        qdim_by_wind: xr.DataArray,
        case_names: list[str],
        wind_sectors: list[str],
    ):
        self.qdim_by_wind = qdim_by_wind
        self.case_names = case_names
        self.wind_sectors = wind_sectors

    def plot_case(self, ax: Axes, case_name: str):
        max_rank = 0

        def draw_sector(position: int, wind_sector: str):
            nonlocal max_rank
            series = self.qdim_by_wind.sel(
                {dn.case_name: case_name, dn.wind_sector: wind_sector}
            ).dropna(dim=dn.space_ix)
            cascade = np.sort(series.values)[::-1]
            ranks = np.arange(1, len(cascade) + 1)
            max_rank = max(max_rank, len(cascade))
            ax.plot(
                ranks,
                cascade,
                marker="o",
                color=WIND_SECTOR_COLOR.get(wind_sector, class_color(position)),
            )

        for position, wind_sector in enumerate(self.wind_sectors):
            draw_sector(position, wind_sector)
        ax.set_xticks(np.arange(1, max_rank + 1))
        ax.set_xlabel("zone rank (best-ventilated first)")
        ax.set_ylabel("q_dim")
        ax.set_title(case_name)

    def make_plot(self) -> Figure:
        fig, axes = plt.subplots(
            nrows=1,
            ncols=len(self.case_names),
            figsize=(7 * len(self.case_names), 5),
            sharey=True,
            squeeze=False,
        )
        for ax, case_name in zip(axes[0], self.case_names):
            self.plot_case(ax, case_name)

        handles = [
            Line2D(
                [],
                [],
                marker="o",
                color=WIND_SECTOR_COLOR.get(wind_sector, class_color(position)),
                label=wind_sector,
            )
            for position, wind_sector in enumerate(self.wind_sectors)
        ]
        fig.legend(
            handles=handles,
            title="Wind Sector",
            loc="lower center",
            bbox_to_anchor=(0.5, -0.1),
            ncol=len(self.wind_sectors),
            frameon=False,
        )
        fig.tight_layout()
        return fig


@dataclass
class CrossValidationReport:
    fold_scores: list[float]

    @property
    def mean(self):
        return statistics.mean(self.fold_scores)

    @property
    def spread(self):
        if len(self.fold_scores) < 2:
            return 0.0
        return statistics.stdev(self.fold_scores)

    def report(self):
        logger.info(
            f"Cross-validation balanced accuracy: {self.mean:.3f} ± {self.spread:.3f} "
            f"over {len(self.fold_scores)} folds"
        )

    def plot_folds(self, ax: Axes):
        folds = list(range(1, len(self.fold_scores) + 1))
        ax.bar(folds, self.fold_scores)
        ax.axhline(self.mean, linestyle="--", color=INK, label=f"mean {self.mean:.3f}")
        ax.set_xlabel("fold")
        ax.set_ylabel("val balanced acc")
        ax.set_ylim(0, 1)
        ax.set_xticks(folds)
        ax.set_title("cross-validation")
        ax.legend()

    def make_plot(self) -> Figure:
        fig, ax = plt.subplots(figsize=(7, 5))
        self.plot_folds(ax)
        fig.tight_layout()
        return fig
