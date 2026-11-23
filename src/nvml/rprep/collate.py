from pathlib import Path
from typing import NamedTuple

import h5py
import numpy as np
import polars as pl
import xarray as xr
from loguru import logger

from nvml import dn
from nvml.flearn.graphs.labels import LabelSchemeNames
from nvml.flearn.kernel.interfaces import KernelNames
from nvml.flearn.pca.make_pca import PCAId
from nvml.nvml.df_names import dfc, direction_map
from nvml.xr_helpers import WindDirectionBinNames


class RegDataId(NamedTuple):
    wind_dir: WindDirectionBinNames
    kernel: KernelNames
    label_scheme: LabelSchemeNames
    num_pcs: int


class RegDataPaths(NamedTuple):
    features_const: Path
    features_wdep: Path

    @property
    def fc(self):
        return pl.read_parquet(self.features_const)

    @property
    def fw(self):
        return pl.read_parquet(self.features_wdep)


def prepare_core_feature_df(rdp: RegDataPaths, wind_dir: WindDirectionBinNames):
    if wind_dir in direction_map.keys():
        long_card_drn = direction_map.get(wind_dir)
        wdf = rdp.fw.filter(pl.col(dfc.wind_sector) == long_card_drn)
        df = rdp.fc.join(wdf, on=dfc.case_name)
    else:
        df = rdp.fc

    return df


def prepare_pca_df(pca_path: Path, pca_id: PCAId, n: int):
    f = h5py.File(pca_path, "r")
    arr = np.array(f[pca_id.name][dn.components])  # pyright: ignore[reportIndexIssue]

    pca = arr[:, :n]
    pcns = [f"pc_{i}" for i in range(n)]
    pca_df = pl.DataFrame(pca, schema=pcns)
    return pca_df


def prepare_qoi_df(qoi_path: Path):
    ds = xr.load_dataset(qoi_path)
    data = ds.groupby(dfc.wind_sector).median(dim=dn.datetime).to_dataframe()
    df = pl.from_pandas(data, include_index=True)
    return df


def collate_data(
    feature_df: pl.DataFrame,
    pca_df: pl.DataFrame,
    qoi_df: pl.DataFrame,
    wind_dir: WindDirectionBinNames,
):
    df2 = pl.concat([feature_df, pca_df], how="horizontal")

    return df2.join(
        qoi_df.filter(pl.col(dfc.wind_sector) == wind_dir), on=dfc.case_name
    )


class DataSpec(NamedTuple):
    fexp_const: Path
    fexp_wdep: Path
    pca: Path
    qoi: Path
    wind_drn: WindDirectionBinNames
    kernel: KernelNames
    label_scheme: LabelSchemeNames
    num_pcs: int

    @property
    def reg_dp(self):
        return RegDataPaths(self.fexp_const, self.fexp_wdep)

    @property
    def pca_id(self):
        return PCAId(self.kernel, self.label_scheme)


def collate_data_from_files(dsp: DataSpec):
    logger.info("Loading data...")
    fexp_df = prepare_core_feature_df(dsp.reg_dp, dsp.wind_drn)
    pca_df = prepare_pca_df(dsp.pca, dsp.pca_id, dsp.num_pcs)
    qoi_df = prepare_qoi_df(dsp.qoi)
    logger.info("Done loading data")
    df = collate_data(fexp_df, pca_df, qoi_df, dsp.wind_drn)
    return df
