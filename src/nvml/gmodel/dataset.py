from plyze.flow_graph.create.main import make_flow_graph
from torch_geometric.utils import from_networkx

from nvml.cli.config import MakeConfig

# TODO: create a dataset that matches with the pytorch_geometric interface based on plyze.
# will probably have multiple datasets based on the different models want to run
#
# need to: move temp data over to scratch, creaate a project config so remove scratch name from repo..
# read in the graphs as flow graphs
# associate each graph with a label based on current algo
# trim down to the data needed..
#
#
# research: best way to compare different graph data vs compare different models vs compare different hyperparams..
# finish reading explainer epaper which probably details how can know if exolanation is good


# from path to pytorch data
def graph_to_torch_data(cfg: MakeConfig, case_name: str):
    G = make_flow_graph(cfg.make_case_data(case_name), cfg.cardinal_expansion_factor)
    torch_data = from_networkx(G)
    return torch_data
