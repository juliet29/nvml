from cyclopts import App
from plyze import FlowGraphModel
from plyze.metrics.helpers.flow_paths import create_flow_paths

from nvml.cli.config import CONFIGS_DICT
from nvml.constants import DataNames
from nvml.flow_paths.main import PathGraph, get_relevant_edges, prep_edge_qdim
from nvml.io import get_ambient_data_as_ds
from nvml.qdim.wind import add_wind_sector_coord

flow = App("flow")


cfg = CONFIGS_DICT["jun24"]
case_name = cfg.get_one_case(0)


@flow.command()
def fb():
    idf, sql = cfg.make_case_data(case_name)
    ad = get_ambient_data_as_ds(sql).pipe(add_wind_sector_coord)
    adw = ad.sel({DataNames.wind_sector: "N"})
    return adw
    # return adw
    # return ad.wind_sector.values


@flow.command()
def fc():
    # get relevant edges
    graph = cfg.make_graph_path(case_name)
    G = FlowGraphModel.read(graph)
    all_paths = create_flow_paths(G)
    path_graphs = [PathGraph.create(i) for i in all_paths]
    se = get_relevant_edges(G, path_graphs)

    # get wind velocity for wind_dir
    _, sql = cfg.make_case_data(case_name)
    ad = get_ambient_data_as_ds(sql).pipe(add_wind_sector_coord)
    adw = ad.sel({DataNames.wind_sector: "N"})[DataNames.wind_speed]

    q_dim = prep_edge_qdim(se[0], adw)
    return q_dim
