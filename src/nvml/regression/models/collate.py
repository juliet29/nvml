from pathlib import Path
from typing import NamedTuple

import h5py
import numpy as np
import polars as pl
import xarray as xr

from nvml import dn
from nvml.qdim.wind import WindDirectionBinNames
from nvml.regression.models.interfaces import dfc, direction_map
from nvml.wl.graphs.labels import LabelSchemeNames
from nvml.wl.kernel.interfaces import KernelNames
from nvml.wl.pca.pca_data import PCAId


class RegDataId(NamedTuple):
    wind_dir: WindDirectionBinNames
    kernel: KernelNames
    label_scheme: LabelSchemeNames
    num_pcs: int

    @property
    def pca_id(self):
        return PCAId(self.kernel, self.label_scheme)


class RegDataPaths(NamedTuple):
    features_const: Path
    features_wdep: Path
    pca: Path

    @property
    def fc(self):
        return pl.read_parquet(self.features_const)

    @property
    def fw(self):
        return pl.read_parquet(self.features_wdep)

    def pc(self, reg_data_id: RegDataId):
        f = h5py.File(self.pca, "r")
        arr = np.array(f[reg_data_id.pca_id.name][dn.components])  # pyright: ignore[reportIndexIssue]
        return arr


def prepare_qoi_df(qoi_path: Path):
    ds = xr.load_dataset(qoi_path)
    data = ds.groupby(dfc.wind_sector).median(dim=dn.datetime).to_dataframe()
    df = pl.from_pandas(data, include_index=True)
    return df


def collate_data(rdp: RegDataPaths, rdid: RegDataId, qoi_df: pl.DataFrame):
    # wind dir dependent features
    if rdid.wind_dir in direction_map.keys():
        long_card_drn = direction_map.get(rdid.wind_dir)
        wdf = rdp.fw.filter(pl.col(dfc.wind_sector) == long_card_drn)
        df = rdp.fc.join(wdf, on=dfc.case_name)
    else:
        df = rdp.fc

    # get the pcas
    pca = rdp.pc(rdid)[:, : rdid.num_pcs]  # index of pca is index into cfg..?
    pcns = [f"pc_{i}" for i in range(rdid.num_pcs)]
    pca_df = pl.DataFrame(pca, schema=pcns)
    df2 = pl.concat([df, pca_df], how="horizontal")

    # get the qois at the  wind dir of interest
    return df2.join(
        qoi_df.filter(pl.col(dfc.wind_sector) == rdid.wind_dir), on=dfc.case_name
    )
