from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures import Future

from cyclopts import App
from loguru import logger
from plyze import FlowGraphModel
from plyze.flow_graph.create.main import make_flow_graph
from rich.pretty import pretty_repr
from tqdm import tqdm

from nvml.cli.config.config import MakeConfig
from nvml.cli.config.cstore import CONFIGS_DICT, config_names
from nvml.logs import silence_plan2eplus_logging
from nvml.utils import default_worker_count

create = App("create")


def build_flow_graph_for_case(config: MakeConfig, case: str):
    case_data = config.make_case_data(case)
    json_file_path = config.make_graph_path(case)
    G = make_flow_graph(case_data, config.cardinal_expansion_factor)
    FlowGraphModel.write(G, json_file_path, config.data_folder_name)


@create.command()
def make_flow_graphs(
    cptr: config_names, workers: int | None = None, skip_existing: bool = False
):
    config = CONFIGS_DICT[cptr]

    cases = config.case_names
    if skip_existing:
        cases = [case for case in cases if not config.make_graph_path(case).exists()]
        skipped_count = len(config.case_names) - len(cases)
        logger.info(f"Skipping {skipped_count} cases with existing graphs")

    failures = []
    max_workers = workers if workers is not None else default_worker_count()
    with ProcessPoolExecutor(
        max_workers=max_workers, initializer=silence_plan2eplus_logging
    ) as executor:
        case_by_future = {
            executor.submit(build_flow_graph_for_case, config, case): case
            for case in cases
        }

        def record(completed: Future[None]):
            case = case_by_future[completed]
            try:
                completed.result()
            except Exception as e:
                failures.append({case: f"{e}"})

        for completed in tqdm(
            as_completed(case_by_future),
            total=len(case_by_future),
            desc="flow graphs",
        ):
            record(completed)

    if failures:
        logger.error(f"Failed cases summary: {pretty_repr(failures)}")
