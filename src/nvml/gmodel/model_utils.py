import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from loguru import logger
from torch_geometric.loader import DataLoader

from nvml.cli.config import MakeConfig
from nvml.constants import FileNames
from nvml.gmodel.dataset import FlowGraphDataset
from nvml.gmodel.dataset_interfaces import ClusterModelParams
from nvml.gmodel.model import GCN
from nvml.gmodel.model_interfaces import (
    GraphModelParams,
    ModelAndDetails,
    ModelTracker,
    SplitDataLoaders,
)


def load_data(
    cfg: MakeConfig,
    save_loc: Path,
    model_params: ClusterModelParams,
):
    ds = FlowGraphDataset(cfg, save_loc)
    ds.cluster(model_params)
    return ds


def split_data(
    ds: FlowGraphDataset,
    batch_size: int,
    n_train: int,
):
    assert isinstance(ds, FlowGraphDataset)
    # breaks potential bias due to similarities in adjacent data
    dss = ds.shuffle()

    train_ds = dss[:n_train]
    test_ds = dss[n_train:]

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return SplitDataLoaders(train_loader, test_loader)


def init_model(graph_params: GraphModelParams):
    """
    hidden_channels: != batch_size: usually a power of 2, 2^4=16, 2^5=32, ...
    """
    model = GCN(graph_params)

    # 3. Define loss function and optimizer

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    return ModelAndDetails(model, criterion, optimizer)


# even more of a util
# TODO: split utils and runners
def log_progress(epoch, epochs, mt: ModelTracker, elapsed: float):
    if (epoch + 1) % 10 == 0:
        logger.info(
            f"Epoch [{epoch + 1}/{epochs}] loss: {mt.avg_loss:.4f} "
            f"acc: {mt.accuracy:.3f} time: {elapsed * 1000:.1f}ms"
        )


def train_model(
    train_loader: DataLoader, mad: ModelAndDetails, save_loc: Path, epochs: int = 100
):

    def handle_batch(batch):
        logits = mad.model(batch.x, batch.edge_index, batch.batch)
        y = batch.y
        loss = mad.criterion(logits, y)

        mad.optimizer.zero_grad()
        loss.backward()
        mad.optimizer.step()

        mt.update_total_loss(loss, y)
        mt.update_correct(logits, y)
        mt.update_seen(y)

    mad.model.train()
    mt = ModelTracker()

    for epoch in range(epochs):
        mt.reset()
        start = time.perf_counter()
        for batch in train_loader:
            handle_batch(batch)
        log_progress(epoch, epochs, mt, time.perf_counter() - start)

    mad.save_model_state(save_loc)


def load_model_state(save_loc: Path, graph_params: GraphModelParams):
    p = save_loc / "models" / FileNames.gnn
    state_dict = torch.load(p, weights_only=True)

    model = GCN(graph_params)
    model.load_state_dict(state_dict)
    return model


def inspect_model(model: GCN):
    # 5. Inspect learned parameters
    [w, b] = model.parameters()
    print(f"Learned Weight: {w[0][0].item():.4f}, Learned Bias: {b[0].item():.4f}")


@torch.no_grad()
def evaluate(test_loader: DataLoader, model: GCN):

    model.eval()
    mt = ModelTracker()
    for batch in test_loader:
        logits = model(batch.x, batch.edge_index, batch.batch)
        y = batch.y
        mt.update_correct(logits, y)
        mt.update_seen(y)

    logger.info(f"Test accuracy: {mt.accuracy:.3f}")
    return mt.accuracy


def predict_with_model():
    pass
