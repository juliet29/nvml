from dataclasses import dataclass

import numpy as np
import polars as pl
import xarray as xr
from plyze.flow_graph.interfaces import ZoneNodeData
from plyze.qoi.xarray_helpers import CardinalNameLiteral

from nvml import dn
from nvml._study.config.config import MakeConfig
from nvml.io import get_ambient_ds_from_cfg, read_graph
from nvml.xr_helpers import WindPolarsEnum, add_wind_sector_coord


def downsample(da: xr.DataArray, time_period: str = "2h"):
    return da.resample({dn.datetime: time_period}).first()


@dataclass
class MathNames:
    # define directions in note / metadata ()
    q = r"$Q_i [m3/s]$"  # Q_i = flow_in - flow_out
    q_dim = r"$Q_i^* = \frac{Q_i}{A_i U}$"
    t_dim = r"$T_i^* = \frac{\Delta T}{T}$"  # dt = Tint - Text
    Ri = r"$Ri=\frac{\Delta T}{T} \frac{g h}{U^2}$"  # h_o = 1
    theta = r"$\theta$"
    U = r"$U$"

    def __getitem__(self, key):
        return getattr(self, key)

    @classmethod
    def get(cls, name: str):
        return cls().__getattribute__(name)


class BuoyData:
    def __init__(
        self,
        cfg: MakeConfig,
        case_name: str,
        room_name: str,
        facade: CardinalNameLiteral,
        downsample: bool = True,
    ) -> None:

        self.room_name = room_name
        self.facade = facade
        graph = read_graph(cfg.make_graph_path(case_name))
        assert graph

        self.graph = graph
        self.zone_data: ZoneNodeData = self.graph.nodes[room_name][dn.flow_graph_data]
        self.edge_data = self.graph.get_edge_data(self.room_name, self.facade)
        self.ambient_data = add_wind_sector_coord(get_ambient_ds_from_cfg(cfg))

        # enviro vars..
        # can assign xarray data here too although will be lost with polars transition..
        U = self.ambient_data[dn.wind_speed]
        self.U = U.where(U != 0, other=np.nan)
        self.T_ext = self.ambient_data[dn.t_out]
        self.theta = self.ambient_data[dn.wind_sector]

        # TODO: should initialize all the other values...

    def calc_temp(self):
        self.t_diff = self.zone_data.temperature - self.T_ext
        self.t_dim = self.t_diff / self.T_ext

    def calc_flow_rate(self):
        data = self.edge_data
        self.q = data.flow_in - data.flow_out

    def calc_dim_flow_rate(self):
        self.calc_flow_rate()
        area = self.edge_data.surface_area
        self.q_dim = self.q / (area * self.U)

    def calc_ri(self):
        self.calc_temp()
        G = 9.81  # TODO: get from constant lib
        ri = self.t_dim * (G / self.U**2)
        self.Ri = ri

    def init_df(self):
        return pl.DataFrame(
            {"theta": self.theta.values}, schema_overrides={"theta": WindPolarsEnum}
        )

    def def_df(self):

        self.calc_dim_flow_rate()
        self.calc_temp()
        self.calc_ri()

        df = self.init_df()

        df2 = df.with_columns(
            **{
                dn.q_dim: self.q_dim.values,
                dn.t_dim: self.t_dim.values,
                dn.U: self.U.values,
                dn.Ri: self.Ri.values,
            }
        ).sort(by=dn.theta)

        x = pl.col(dn.Ri)
        df3 = df2.filter(x.is_between(x.quantile(0.05), x.quantile(0.95)))

        return df3
