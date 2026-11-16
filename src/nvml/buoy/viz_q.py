from plyze.flow_graph.interfaces import ZoneNodeData

from nvml import dn
from nvml._study.config.config import MakeConfig
from nvml.io import read_graph


def plot_temp_days(cfg: MakeConfig, case_name: str, room_name: str):
    graph = read_graph(cfg.make_graph_path(case_name))
    assert graph
    data: ZoneNodeData = graph.nodes[room_name][dn.flow_graph_data]

    # TODO: need to get the right edges
    #
    #
    # .
