from cyclopts import App
from plyze import FlowGraphModel
from plyze.metrics.helpers.flow_paths import create_flow_paths

from nvml.cli.config import CONFIGS_DICT
from nvml.flow_paths.main import PathGraph, get_relevant_edges

flow = App("flow")


cfg = CONFIGS_DICT["jun24"]


@flow.command()
def fc():
    path = cfg.make_json_path(cfg.get_one_case())
    G = FlowGraphModel.read(path)
    all_paths = create_flow_paths(G)
    path_graphs = [PathGraph.create(i) for i in all_paths]
    se = get_relevant_edges(G, path_graphs)
    # edges = get_unique_edges(path_graphs)
    # se = set(edges)
    return se
