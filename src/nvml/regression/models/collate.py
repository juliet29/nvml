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
    qoi: Path
    pca: Path

    @property
    def fc(self):
        return pl.read_parquet(self.features_const)

    @property
    def fw(self):
        return pl.read_parquet(self.features_wdep)

    @property
    def qo(self):
        return xr.load_dataset(self.qoi)

    def pc(self, reg_data_id: RegDataId):
        f = h5py.File(self.pca, "r")
        arr = np.array(f[reg_data_id.pca_id])
        return arr


def collate_data(rdp: RegDataPaths, rdid: RegDataId):
    # get the wind directions for rdid

    long_card_drn = direction_map.get(rdid.wind_dir)
    wdf = rdp.fw.filter(pl.col(dfc.wind_sector) == long_card_drn)

    df = rdp.fc.join(wdf, on=dfc.wind_sector)

    # get the qois -> time median, wind dir of interest
    qoi = (
        rdp.qo.sel({dn.wind_sector: rdid.wind_dir}).median(dim=dn.datetime).to_pandas()
    )
    qoi_df = pl.from_pandas(qoi, include_index=True)
    df2 = df.join(qoi_df)

    # get the pcas
    pca = rdp.pc(rdid)[:, : rdid.num_pcs]  # index of pca is index into cfg..?
