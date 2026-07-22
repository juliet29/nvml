from cyclopts import App

from nvml.cli.studies.helpers import CASE_NAME, cfg
from nvml.gmodel.dataset import graph_to_torch_data

gmod = App("gmod")


def fc():
    return graph_to_torch_data(cfg, CASE_NAME)
