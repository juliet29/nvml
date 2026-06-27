from pathlib import Path

from cyclopts import App
from loguru import logger
from plyze import CaseData, FlowGraphModel
from plyze.flow_graph.create.main import make_flow_graph
from rich.pretty import pretty_repr

from nvml.cli.config import CONFIGS_DICT

create = App("create")


@create.command()
def make_flow_graphs(cptr: str):
    config = CONFIGS_DICT[cptr]

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
            continue

        logger.success(f"Finished making {case}")

    if failures:
        logger.error(f"Failed cases summary: {pretty_repr(failures)}")
