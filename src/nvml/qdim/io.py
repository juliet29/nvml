from pathlib import Path

from plyze import FlowGraphModel
from plyze.flow_graph.create.main import make_ambient_data
from plyze.qoi_flow_graph.zone_data import collate_ambient_data, collate_zone_data


# TODO: move io stuff elsewhere
def graph_to_ds(path: Path):
    def handle(path: Path):
        G = FlowGraphModel.read(path)
        ds = collate_zone_data(G)
        return ds

    return handle(path)


def get_ambient_data_as_ds(path: Path):
    """path is eplusout.sql"""
    res = make_ambient_data(path)
    return collate_ambient_data(res)
