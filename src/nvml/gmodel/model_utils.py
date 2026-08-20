from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from loguru import logger
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch_geometric.loader import DataLoader

from nvml.cli.config.config import MakeConfig
from nvml.constants import FileNames
from nvml.gmodel.dataset import FlowGraphDataset
from nvml.gmodel.dataset_interfaces import ClusterModelParams
from nvml.gmodel.model import HeterogGCN
from nvml.gmodel.model_interfaces import (
    GraphModelParams,
    ModelAndDetails,
    ModelTracker,
    SplitDataLoaders,
)
from nvml.gmodel.viz import EpochMetrics, TrainingWatcher


def load_data(
    cfg: MakeConfig,
    save_loc: Path,
    model_params: ClusterModelParams,
):
    ds = FlowGraphDataset(cfg, save_loc)
    ds.cluster(model_params)
    return ds[ds.retained_indices]


def split_data(
    ds: FlowGraphDataset,
    batch_size: int,
    seed: int = 1204,
):
    labels = [int(ds[i].y) for i in range(len(ds))]

    train_idx, holdout_idx = train_test_split(
        range(len(ds)), test_size=0.30, stratify=labels, random_state=seed
    )
    val_idx, test_idx = train_test_split(
        holdout_idx,
        test_size=0.50,
        stratify=[labels[i] for i in holdout_idx],
        random_state=seed,
    )

    train_loader = DataLoader(ds[train_idx], batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(ds[val_idx], batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(ds[test_idx], batch_size=batch_size, shuffle=False)

    return SplitDataLoaders(train_loader, val_loader, test_loader)


def init_model(graph_params: GraphModelParams):
    """
    hidden_channels: != batch_size: usually a power of 2, 2^4=16, 2^5=32, ...
    """
    model = HeterogGCN(graph_params)

    # 3. Define loss function and optimizer

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    return ModelAndDetails(model, criterion, optimizer)


def train_model(
    train_loader: DataLoader,
    mad: ModelAndDetails,
    save_loc: Path,
    epochs: int = 100,
    val_loader: DataLoader | None = None,
    watcher: TrainingWatcher | None = None,
):
    watcher = watcher or TrainingWatcher()
    mt = ModelTracker()

    def handle_batch(batch):
        logits = mad.model(batch.x_dict, batch.edge_index_dict, batch.batch_dict)
        y = batch.y
        loss = mad.criterion(logits, y)

        mad.optimizer.zero_grad()
        loss.backward()
        mad.optimizer.step()

        mt.update_total_loss(loss, y)
        mt.update_correct(logits, y)
        mt.update_seen(y)

    for epoch in range(epochs):
        mad.model.train()
        mt.reset()
        for batch in train_loader:
            handle_batch(batch)

        val_accuracy = evaluate(val_loader, mad.model, log=False) if val_loader else None
        watcher.record(EpochMetrics(epoch + 1, mt.avg_loss, mt.accuracy, val_accuracy))

    mad.save_model_state(save_loc)
    watcher.report_summary()
    return watcher


def cross_validate(
    ds: FlowGraphDataset,
    graph_params: GraphModelParams,
    batch_size: int,
    save_loc: Path,
    n_folds: int = 5,
    epochs: int = 100,
    seed: int = 1204,
):
    labels = [int(ds[i].y) for i in range(len(ds))]
    folds = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    fold_accuracies = []
    for fold, (train_idx, val_idx) in enumerate(folds.split(range(len(ds)), labels)):
        logger.info(f"Fold {fold + 1}/{n_folds}")
        train_loader = DataLoader(
            ds[train_idx.tolist()], batch_size=batch_size, shuffle=True
        )
        val_loader = DataLoader(ds[val_idx.tolist()], batch_size=batch_size)

        mad = init_model(graph_params)
        train_model(train_loader, mad, save_loc, epochs, val_loader=val_loader)
        fold_accuracies.append(evaluate(val_loader, mad.model, log=False))

    return fold_accuracies


def load_model_state(save_loc: Path, graph_params: GraphModelParams):
    p = save_loc / "models" / FileNames.gnn
    state_dict = torch.load(p, weights_only=True)

    model = HeterogGCN(graph_params)
    model.load_state_dict(state_dict)
    return model


def inspect_model(model: HeterogGCN):
    # 5. Inspect learned parameters
    [w, b] = model.parameters()
    print(f"Learned Weight: {w[0][0].item():.4f}, Learned Bias: {b[0].item():.4f}")


@torch.no_grad()
def evaluate(test_loader: DataLoader, model: HeterogGCN, log: bool = True):

    model.eval()
    mt = ModelTracker()
    for batch in test_loader:
        logits = model(batch.x_dict, batch.edge_index_dict, batch.batch_dict)
        y = batch.y
        mt.update_correct(logits, y)
        mt.update_seen(y)

    if log:
        logger.info(f"Test accuracy: {mt.accuracy:.3f}")
    return mt.accuracy


def predict_with_model():
    pass
