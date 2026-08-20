import statistics
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
from loguru import logger
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from nvml.utils import save_mpl_fig


@dataclass
class EpochMetrics:
    epoch: int
    train_loss: float
    train_accuracy: float
    val_accuracy: float | None = None


@dataclass
class TrainingWatcher:
    log_every: int = 10
    save_path: Path | None = None
    save_every: int = 5
    show_live: bool = False
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

    def save_live_figure(self, save_path: Path):
        fig = VisualizeTraining(self).make_plot()
        save_mpl_fig(fig, save_path)
        plt.close(fig)

    def show_live_figure(self):
        fig = VisualizeTraining(self).make_plot()
        plt.show()
        plt.close(fig)

    def report_latest(self):
        m = self.history[-1]
        val = "" if m.val_accuracy is None else f" val_acc: {m.val_accuracy:.3f}"
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
        validated = [m for m in self.history if m.val_accuracy is not None]
        if validated:
            best = max(validated, key=lambda m: m.val_accuracy or 0.0)
            logger.info(f"Best val_acc {best.val_accuracy:.3f} at epoch {best.epoch}")


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
        validated = [m for m in history if m.val_accuracy is not None]
        if validated:
            ax.plot(
                [m.epoch for m in validated],
                [m.val_accuracy for m in validated],
                label="val accuracy",
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


@dataclass
class CrossValidationReport:
    fold_accuracies: list[float]

    @property
    def mean(self):
        return statistics.mean(self.fold_accuracies)

    @property
    def spread(self):
        if len(self.fold_accuracies) < 2:
            return 0.0
        return statistics.stdev(self.fold_accuracies)

    def report(self):
        logger.info(
            f"Cross-validation accuracy: {self.mean:.3f} ± {self.spread:.3f} "
            f"over {len(self.fold_accuracies)} folds"
        )

    def plot_folds(self, ax: Axes):
        folds = list(range(1, len(self.fold_accuracies) + 1))
        ax.bar(folds, self.fold_accuracies)
        ax.axhline(self.mean, linestyle="--", color="black", label=f"mean {self.mean:.3f}")
        ax.set_xlabel("fold")
        ax.set_ylabel("val accuracy")
        ax.set_ylim(0, 1)
        ax.set_xticks(folds)
        ax.set_title("cross-validation")
        ax.legend()

    def make_plot(self) -> Figure:
        fig, ax = plt.subplots(figsize=(7, 5))
        self.plot_folds(ax)
        fig.tight_layout()
        return fig
