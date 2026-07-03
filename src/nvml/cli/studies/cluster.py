from cyclopts import App

from nvml.cli.config import CONFIGS_DICT

cluster = App("cluster")

cfg = CONFIGS_DICT["test_cluster"]


@cluster.command()
def make_one_graph():
    case_data = cfg.get_one_case_data()
    # json_file = cfg.make_json_path(cfg.get_one_case())
    G = make_flow_graph(case_data, cfg.cardinal_expansion_factor)
    return G
    # FlowGraphModel.write(G, json_file_path, cfg.data_folder_name)
