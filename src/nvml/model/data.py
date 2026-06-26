from pathlib import Path

import numpy as np
import polars as pl
import torch
from loguru import logger
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

from nvml.constants import DataNames as dn
from nvml.model.interfaces import PolarsData


def collect_metrics_data(path: Path):
    df = pl.read_csv(path)
    return df

    # df_agg = df.select(c)
    # logger.debug(csv)


def collect_qoi_data(path: Path):
    dnq = dn.variables
    df = pl.read_csv(path)
    # take the median value across the sample times, the sum across the values for zones to produce one number per case
    df_agg = (
        df.group_by(dnq.case_name, dnq.space_name)
        .agg(pl.col(dnq.zone_dimless_flow).median())
        .group_by(dnq.case_name)
        .agg(pl.col(dnq.zone_dimless_flow).sum())
        # .select(dnq.zone_dimless_flow)
    )
    return df_agg


def arrange_data(metrics_path: Path, qois_path: Path):
    dnq = dn.variables
    metrics = collect_metrics_data(metrics_path)
    qoi = collect_qoi_data(qois_path)
    df = metrics.join(qoi, on=dnq.case_name)

    X = df.select(pl.exclude(dnq.case_name, dnq.zone_dimless_flow))
    y = df.select(dnq.zone_dimless_flow)
    return PolarsData(X, y)


class NVDataset(Dataset):
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        mean: np.ndarray,
        std: np.ndarray,
        edges: np.ndarray,
    ):
        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.float32)
        self.mean = torch.as_tensor(mean, dtype=torch.float32)
        self.std = torch.as_tensor(std, dtype=torch.float32)
        self.edges = torch.as_tensor(edges, dtype=torch.float32).ravel()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        X = (self.X[i] - self.mean) / self.std
        # right=True matches np.digitize's default bin semantics
        label = torch.bucketize(self.y[i], self.edges, right=True).squeeze(-1)
        return X, label


def create_train_test_dataset(pd: PolarsData, seed: int):
    # ---- Split
    # provisional labels only drive the stratified split; final labels use train edges
    y = pd.ynp
    provisional = np.digitize(y.ravel(), np.quantile(y, [1 / 3, 2 / 3]))
    X_train, X_test, y_train, y_test = train_test_split(
        pd.Xnp, y, train_size=0.8, random_state=seed, stratify=provisional
    )
    logger.info("Split data")

    # --- Get edges for labels
    edges = np.quantile(y_train, [1 / 3, 2 / 3])
    logger.info(f"Tercile edges from y_train: {edges}")

    # --- Get scaling info for features
    scaler = StandardScaler()
    scaler.fit(X_train)
    mean, std = scaler.mean_, scaler.scale_
    logger.info(f"Fit scaler with X_train -- mean: {mean} --- std: {std}")

    # --- Create datasets
    train_ds = NVDataset(X_train, y_train, mean, std, edges)  # pyright: ignore[reportArgumentType]
    test_ds = NVDataset(X_test, y_test, mean, std, edges)  # pyright: ignore[reportArgumentType]

    return train_ds, test_ds
