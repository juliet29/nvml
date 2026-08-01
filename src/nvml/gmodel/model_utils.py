import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from loguru import logger
from torch.utils.data import DataLoader

from nvml.cli.config import MakeConfig
from nvml.gmodel.dataset import FlowGraphDataset
from nvml.gmodel.interfaces import ModelAndDetails, ModelTracker, SplitDataLoaders
from nvml.gmodel.model import GCN
from nvml.qdim.wind import WindDirectionBinNames


def prep_data(
    cfg: MakeConfig,
    save_loc: Path,
    wind_sector: WindDirectionBinNames,
    n_clusters: int,
    batch_size: int,
    n_train: int,
):
    ds = FlowGraphDataset(cfg, save_loc)

    # creates and save clustering model that will hold labels for each graph
    ds.cluster(wind_sector, n_clusters)

    # breaks potential bias due to similarities in adjacent data
    ds = ds.shuffle()

    train_ds = ds[:n_train]
    test_ds = ds[n_train:]

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return SplitDataLoaders(train_loader, test_loader)


def init_model(hidden_channels: int):
    """
    hidden_channels: != batch_size
    """
    model = GCN(hidden_channels)

    # 3. Define loss function and optimizer

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    return ModelAndDetails(model, criterion, optimizer)


def log_progress(epoch, epochs, mt: ModelTracker, elapsed: float):
    if (epoch + 1) % 10 == 0:
        logger.info(
            f"Epoch [{epoch + 1}/{epochs}] loss: {mt.avg_loss:.4f} "
            f"acc: {mt.accuracy:.3f} time: {elapsed * 1000:.1f}ms"
        )


def train_model(train_ds: FlowGraphDataset, mad: ModelAndDetails, epochs: int = 100):

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
def evaluate(test_ds: FlowGraphDataset, mad: ModelAndDetails):
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
