from typing import NamedTuple

import polars as pl
import torch


class PolarsData(NamedTuple):
    X: pl.DataFrame
    y: pl.DataFrame

    @property
    def Xnp(self):
        return self.X.to_numpy()

    @property
    def ynp(self):
        return self.y.to_numpy()


class TorchData(NamedTuple):
    X: torch.Tensor
    y: torch.Tensor
