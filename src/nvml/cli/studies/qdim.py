from cyclopts import App
from plyze.flow_graph.create.main import make_ambient_data

from nvml.cli.config import CONFIGS_DICT
from nvml.qdim.data import graph_to_df

qdim = App("qdim")


cfg = CONFIGS_DICT["jun24"]


@qdim.command()
def fc():
    case_0 = cfg.case_names[0]
    path = cfg.make_json_path(case_0)
    res = graph_to_df(path)
    sn = res.select("space_names").unique()
    dt = res.select("datetimes").unique()
    return sn, dt


@qdim.command()
def fd():
    res = make_ambient_data(cfg.get_one_case_data().sql)
    return res
    pass
