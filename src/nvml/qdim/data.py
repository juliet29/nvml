from pathlib import Path

from plyze import FlowGraphModel
from plyze.qoi_flow_graph.zone_data import collate_zone_data_to_df


def graph_to_df(path: Path):
    def handle(path: Path):
        G = FlowGraphModel.read(path)
        df = collate_zone_data_to_df(G)
        return df

    return handle(path)
