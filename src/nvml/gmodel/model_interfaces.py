from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import torch
import torch.nn as nn
from loguru import logger
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from utils4plans.io import make_dir

from nvml.constants import FileNames
from nvml.gmodel.dataset import FlowGraphDataset


class GraphModelParams(NamedTuple):
    hidden_channels: int
    num_node_features: int
    num_classes: int

    @classmethod
    def make(cls, hidden_channels: int, ds: FlowGraphDataset):
        return cls(
            hidden_channels=hidden_channels,
            num_node_features=ds.num_node_features,
            num_classes=ds.num_classes,
        )


class SplitDataLoaders(NamedTuple):
    train: DataLoader
    test: DataLoader


class ModelAndDetails(NamedTuple):
    model: nn.Module
    criterion: _Loss
    optimizer: Optimizer

    def save_model_state(self, save_loc: Path):
        p = save_loc / "models" / FileNames.gnn
        make_dir(p)

        torch.save({"model": self.model.state_dict(), "optimizer": self.optimizer}, p)
        logger.info(f"Saved model to {p}")


@dataclass
class ModelTracker:
    total_loss: float = 0.0
    correct: int = 0
    seen: int = 0

    def reset(self):
        self.total_loss = 0.0
        self.correct = 0
        self.seen = 0

    def update_total_loss(self, loss, y):
        self.total_loss += loss.item() * len(y)

    def update_correct(self, logits, y):
        self.correct += (logits.argmax(1) == y).sum().item()

    def update_seen(self, y):
        self.seen += len(y)

    @property
    def avg_loss(self):
        return self.total_loss / self.seen

    @property
    def accuracy(self):
        return self.correct / self.seen
