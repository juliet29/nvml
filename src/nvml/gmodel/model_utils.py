from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from loguru import logger
from rich.console import Console
from rich.pretty import pretty_repr
from rich.prompt import Confirm
from rich.table import Table
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch_geometric.loader import DataLoader
from utils4plans.io import make_dir

from nvml.cli.config.config import MakeConfig
from nvml.gmodel.dataset import FlowGraphDataset
from nvml.gmodel.dataset_interfaces import ClusterModelParams
from nvml.gmodel.model import HeterogGCN
from nvml.gmodel.processing import GModelNames
from nvml.gmodel.model_interfaces import (
    EvalMetrics,
    GraphModelParams,
    ModelAndDetails,
    ModelTracker,
    SplitDataLoaders,
)
from nvml.gmodel.viz import (
    EpochMetrics,
    TrainingWatcher,
    VisualizeCrossValidationConvergence,
)


def confirm_class_balance(labels: list[int]):
    counts = Counter(labels)
    total = len(labels)
    majority = max(counts.values()) / total

    table = Table(title="Class balance")
    table.add_column("class")
    table.add_column("count", justify="right")
    table.add_column("fraction", justify="right")
    for label in sorted(counts):
        table.add_row(str(label), str(counts[label]), f"{counts[label] / total:.3f}")
    table.caption = f"n={total}   majority-class baseline={majority:.3f}"
    Console().print(table)

    if not Confirm.ask("Class balance OK — proceed with training?", default=True):
        raise SystemExit("Aborted: class balance rejected")


def load_data(
    cfg: MakeConfig,
    save_loc: Path,
    model_params: ClusterModelParams,
    seed: int = 1204,
):
    ds = FlowGraphDataset(cfg, save_loc)
    ds.cluster(model_params, seed)
    retained = ds[ds.retained_indices]

    labels = [int(retained[i].y) for i in range(len(retained))]
    confirm_class_balance(labels)
    return retained


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


def compute_class_weights(labels: list[int], num_classes: int):
    counts = Counter(labels)
    total = len(labels)
    weights = [
        total / (num_classes * counts[c]) if counts.get(c) else 0.0
        for c in range(num_classes)
    ]
    return torch.tensor(weights, dtype=torch.float)


def init_model(
    graph_params: GraphModelParams,
    class_weights: torch.Tensor | None = None,
    lr: float = 1e-3,
):
    model = HeterogGCN(graph_params)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    return ModelAndDetails(model, criterion, optimizer)


def train_model(
    train_loader: DataLoader,
    mad: ModelAndDetails,
    save_loc: Path,
    tag: str,
    epochs: int = 100,
    val_loader: DataLoader | None = None,
    watcher: TrainingWatcher | None = None,
):
    watcher = watcher or TrainingWatcher()
    mt = ModelTracker()
    best_val_score = None
    best_model_path = save_loc / "models" / GModelNames.make_best_model_name(tag)

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

    def checkpoint_if_best(val_score: float):
        nonlocal best_val_score
        if best_val_score is None or val_score > best_val_score:
            best_val_score = val_score
            make_dir(best_model_path)
            torch.save(mad.model.state_dict(), best_model_path)

    for epoch in range(epochs):
        mad.model.train()
        mt.reset()
        for batch in train_loader:
            handle_batch(batch)

        val_score = (
            evaluate(val_loader, mad.model, log=False).balanced_accuracy
            if val_loader
            else None
        )
        watcher.record(EpochMetrics(epoch + 1, mt.avg_loss, mt.accuracy, val_score))
        if val_score is not None:
            checkpoint_if_best(val_score)

    mad.save_model_state(save_loc, GModelNames.make_model_name(tag))
    watcher.report_summary()
    return watcher


def evaluate_best_model(
    save_loc: Path,
    graph_params: GraphModelParams,
    tag: str,
    val_loader: DataLoader,
    test_loader: DataLoader,
):
    best_model_path = save_loc / "models" / GModelNames.make_best_model_name(tag)
    model = HeterogGCN(graph_params)
    model.load_state_dict(torch.load(best_model_path, weights_only=True))

    val_metrics = evaluate(val_loader, model, log=True, label="Best model VAL")
    test_metrics = evaluate(test_loader, model, log=True, label="Best model TEST")
    return val_metrics, test_metrics


def cross_validate(
    ds: FlowGraphDataset,
    graph_params: GraphModelParams,
    batch_size: int,
    save_loc: Path,
    tag: str,
    n_folds: int = 5,
    epochs: int = 100,
    lr: float = 1e-3,
    show_live: bool = False,
    seed: int = 1204,
):
    labels = [int(ds[i].y) for i in range(len(ds))]
    folds = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    num_classes = graph_params.num_classes

    fold_scores = []
    fold_watchers = [TrainingWatcher(show_live=show_live) for _ in range(n_folds)]
    out_of_fold_confusion = np.zeros((num_classes, num_classes), dtype=int)

    if show_live:
        def render_live_grid():
            return VisualizeCrossValidationConvergence(fold_watchers).make_plot()

        for watcher in fold_watchers:
            watcher.make_live_figure = render_live_grid

    for fold, (train_idx, val_idx) in enumerate(folds.split(range(len(ds)), labels)):
        logger.info(f"Fold {fold + 1}/{n_folds}")
        train_positions = train_idx.tolist()
        train_labels = [labels[i] for i in train_positions]
        train_loader = DataLoader(
            ds[train_positions], batch_size=batch_size, shuffle=True
        )
        val_loader = DataLoader(ds[val_idx.tolist()], batch_size=batch_size)

        mad = init_model(
            graph_params, compute_class_weights(train_labels, num_classes), lr
        )
        train_model(
            train_loader,
            mad,
            save_loc,
            tag,
            epochs,
            val_loader=val_loader,
            watcher=fold_watchers[fold],
        )
        fold_metrics = evaluate(val_loader, mad.model, log=False)
        fold_scores.append(fold_metrics.balanced_accuracy)
        out_of_fold_confusion += np.array(fold_metrics.confusion)

    confusion = out_of_fold_confusion.tolist()
    logger.info(
        f"Cross-validation out-of-fold confusion (rows=true, cols=pred):\n"
        f"{pretty_repr(confusion)}"
    )
    return fold_scores, fold_watchers, confusion


def load_model_state(save_loc: Path, graph_params: GraphModelParams, tag: str):
    p = save_loc / "models" / GModelNames.make_model_name(tag)
    state_dict = torch.load(p, weights_only=True)

    model = HeterogGCN(graph_params)
    model.load_state_dict(state_dict)
    return model


def load_best_model_state(save_loc: Path, graph_params: GraphModelParams, tag: str):
    p = save_loc / "models" / GModelNames.make_best_model_name(tag)
    state_dict = torch.load(p, weights_only=True)

    model = HeterogGCN(graph_params)
    model.load_state_dict(state_dict)
    return model


def inspect_model(model: HeterogGCN):
    # 5. Inspect learned parameters
    [w, b] = model.parameters()
    print(f"Learned Weight: {w[0][0].item():.4f}, Learned Bias: {b[0].item():.4f}")


@torch.no_grad()
def evaluate(
    loader: DataLoader, model: HeterogGCN, log: bool = True, label: str = "Test"
) -> EvalMetrics:
    model.eval()
    class_labels = list(range(model.lin.out_features))

    y_true, y_pred = [], []
    for batch in loader:
        logits = model(batch.x_dict, batch.edge_index_dict, batch.batch_dict)
        y_true.extend(batch.y.tolist())
        y_pred.extend(logits.argmax(1).tolist())

    metrics = EvalMetrics(
        accuracy=accuracy_score(y_true, y_pred),
        balanced_accuracy=balanced_accuracy_score(y_true, y_pred),
        macro_f1=f1_score(
            y_true, y_pred, labels=class_labels, average="macro", zero_division=0
        ),
        confusion=confusion_matrix(y_true, y_pred, labels=class_labels).tolist(),
    )
    if log:
        metrics.log(label)
    return metrics


def predict_with_model():
    pass
