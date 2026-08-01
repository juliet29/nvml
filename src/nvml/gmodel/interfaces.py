from dataclasses import dataclass
from typing import NamedTuple

import torch.nn as nn
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from nvml.qdim.wind import WindDirectionBinNames


class ClusterModelParams(NamedTuple):
    wind_sector: WindDirectionBinNames
    n_clusters: int


class SplitDataLoaders(NamedTuple):
    train: DataLoader
    test: DataLoader


class ModelAndDetails(NamedTuple):
    model: nn.Module
    criterion: _Loss
    optimizer: Optimizer


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
