import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from matplotlib.axes import Axes
from matplotlib.figure import Figure

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
        ax.imshow(matrix, cmap="Blues")
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
        ax.axvline(best_k, linestyle="--", color="black", label=f"best k={best_k}")
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
        ax.axhline(self.mean, linestyle="--", color="black", label=f"mean {self.mean:.3f}")
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
