from cyclopts import App

from nvml.cli.config import CONFIGS_DICT
from nvml.qdim.data import graph_to_df

qdim = App("qdim")


@qdim.command()
def fc():
    config = CONFIGS_DICT["jun24"]
    case_0 = config.case_names[0]
    path = config.make_json_path(case_0)
    res = graph_to_df(path)
    return res
