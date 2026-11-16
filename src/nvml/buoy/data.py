import xarray as xr
from plyze.flow_graph.interfaces import ZoneNodeData
from plyze.qoi.xarray_helpers import CardinalNameLiteral

from nvml import dn
from nvml._study.config.config import MakeConfig
from nvml.io import get_ambient_ds_from_cfg, read_graph
from nvml.qdim.wind import add_wind_sector_coord


def downsample(da: xr.DataArray, time_period: str = "2h"):
    return da.resample({dn.datetime: time_period}).first()


class BuoyData:
    def __init__(
        self,
        cfg: MakeConfig,
        case_name: str,
        room_name: str,
        downsample: bool = True,
    ) -> None:

        self.room_name = room_name
        graph = read_graph(cfg.make_graph_path(case_name))
        assert graph

        self.graph = graph
        self.zone_data: ZoneNodeData = self.graph.nodes[room_name][dn.flow_graph_data]
        self.ambient_data = add_wind_sector_coord(get_ambient_ds_from_cfg(cfg))

        # enviro vars..
        self.U = self.ambient_data[dn.wind_speed]
        self.T_ext = self.ambient_data[dn.t_out]
        self.theta = self.ambient_data[dn.wind_sector]

    def calc_temp(self):
        self.t_diff = self.zone_data.temperature - self.T_ext
        self.t_dim = self.t_diff / self.T_ext

    def calc_flow_rate(self, facade: CardinalNameLiteral):
        data = self.graph.get_edge_data(self.room_name, facade)
        q = data.flow_in - data.flow_out
        return q

    def calc_dim_flow_rate(self, facade: CardinalNameLiteral):
        data = self.graph.get_edge_data(self.room_name, facade)
        area = data.surface_area
        q = self.calc_flow_rate(facade)
        return q / (area * self.U)
