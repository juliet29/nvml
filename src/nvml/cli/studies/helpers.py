from plyze import FlowGraphModel

from nvml.cli.config import CONFIGS_DICT
from nvml.io import get_ambient_data_as_ds, graph_to_ds
from nvml.qdim.wind import (
    add_wind_sector_coord,
)

cfg = CONFIGS_DICT["jun24"]
CASE_IX = 0
CASE_NAME = cfg.get_one_case(CASE_IX)


def get_graph_path(case_name: str = CASE_NAME):
    path = cfg.make_graph_path(case_name)
    return path


def get_graph(case_name: str = CASE_NAME):
    path = cfg.make_graph_path(case_name)
    G = FlowGraphModel.read(path)
    return G


def get_qoi_ds(case_name: str = CASE_NAME):
    path = cfg.make_graph_path(case_name)
    res = graph_to_ds(path)
    return res


def get_ambient_ds(ix: int = CASE_IX):
    res = get_ambient_data_as_ds(cfg.get_one_case_data(ix).sql)
    res = add_wind_sector_coord(res)
    return res


def get_all_data():
    G = get_graph()
    qoi_ds = get_qoi_ds()
    ambient_ds = get_ambient_ds()
    return G, qoi_ds, ambient_ds
