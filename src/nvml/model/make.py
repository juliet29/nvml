import time
from dataclasses import dataclass
from typing import NamedTuple

import torch
import torch.nn as nn
import torch.optim as optim
from loguru import logger
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from nvml.model.data import NVDataset


class NVClassifier(nn.Module):
    def __init__(
        self,
        in_features: int,
        n_classes: int = 3,
        hidden: int = 32,
        p_drop: float = 0.2,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(),
            nn.Dropout(p_drop),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(p_drop),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x):
        return self.net(x)


class ModelAndDetails(NamedTuple):
    model: nn.Module
    criterion: _Loss
    optimizer: Optimizer


# TODO: should try with sklern tree based models becuase discrete nature of model. Doing this to get a handle on torch before mover to PytorchGeoemtric for graphs
def init_model(train_ds: NVDataset):
    model = NVClassifier(in_features=train_ds.X.shape[1])

    # 3. Define loss function and optimizer

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    return ModelAndDetails(model, criterion, optimizer)


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


def log_progress(epoch, epochs, mt: ModelTracker, elapsed: float):
    if (epoch + 1) % 10 == 0:
        logger.info(
            f"Epoch [{epoch + 1}/{epochs}] loss: {mt.avg_loss:.4f} "
            f"acc: {mt.accuracy:.3f} time: {elapsed * 1000:.1f}ms"
        )


def train_model(train_ds: NVDataset, mad: ModelAndDetails, epochs: int = 100):

    def handle_batch(X, y):
        logits = mad.model(X)
        loss = mad.criterion(logits, y)

        mad.optimizer.zero_grad()
        loss.backward()
        mad.optimizer.step()

        mt.update_total_loss(loss, y)
        mt.update_correct(logits, y)
        mt.update_seen(y)

    loader = DataLoader(train_ds, batch_size=32, shuffle=True)

    mad.model.train()
    mt = ModelTracker()

    for epoch in range(epochs):
        mt.reset()
        start = time.perf_counter()
        for X, y in loader:
            handle_batch(X, y)
        log_progress(epoch, epochs, mt, time.perf_counter() - start)


def inspect_model(mad: ModelAndDetails):
    # 5. Inspect learned parameters
    [w, b] = mad.model.parameters()
    print(f"Learned Weight: {w[0][0].item():.4f}, Learned Bias: {b[0].item():.4f}")


@torch.no_grad()
def evaluate(test_ds: NVDataset, mad: ModelAndDetails):
    loader = DataLoader(test_ds, batch_size=64)

    mad.model.eval()
    mt = ModelTracker()
    for X, y in loader:
        logits = mad.model(X)
        mt.update_correct(logits, y)
        mt.update_seen(y)

    logger.info(f"Test accuracy: {mt.accuracy:.3f}")
    return mt.accuracy


def predict_with_model():
    pass
