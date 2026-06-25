from pathlib import Path

from cyclopts import App
from loguru import logger
from plyze import CaseData, FlowGraphModel
from plyze.flow_graph.create.main import make_flow_graph

from nvml.cli.make.config import MakeConfig

create = App("create")


def make_flow_graphs(config: MakeConfig):
    def make_one_graph(case_data: CaseData, json_file_path: Path):
        G = make_flow_graph(case_data, config.cardinal_expansion_factor)
        FlowGraphModel.write(G, json_file_path, config.data_folder_name)

    failures = []
    for case in config.case_names:
        try:
            cd = config.make_case_data(case)
            jsf = config.make_json_path(case)
            make_one_graph(cd, jsf)
        except Exception as e:
            logger.error(f"Failed to make {case} because: {e}")
            failures.append({case: f"{e}"})
        logger.success(f"Finished making {case}")
