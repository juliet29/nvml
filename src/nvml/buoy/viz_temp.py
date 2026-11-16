from plyze.flow_graph.interfaces import ZoneNodeData

from nvml import dn
from nvml._study.config.config import MakeConfig
from nvml.io import read_graph

DIFF_THRESHOLD = 0.1


def plot_temp_days(cfg: MakeConfig, case_name: str, room_name: str):
    graph = read_graph(cfg.make_graph_path(case_name))
    assert graph
    data: ZoneNodeData = graph.nodes[room_name][dn.flow_graph_data]
    temp = data.__getattribute__(dn.zone_temp)

    dta = temp[dn.datetime].dt
    da = temp.sel({dn.datetime: (dta.day == 15) & (dta.month == 7)})
    da_diff = da.diff(dim=dn.datetime)

    small_diff_times = da_diff.where(
        (da_diff < 0.1) & (da_diff > -0.1), drop=True
    ).datetimes
    low_diff_da = da.sel({dn.datetime: small_diff_times})

    return graph, da, da_diff, low_diff_da
