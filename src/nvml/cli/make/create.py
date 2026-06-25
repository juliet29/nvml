from cyclopts import App
from plyze import CaseData, FlowGraphModel
from plyze.flow_graph.create.main import make_flow_graph

create = App("create")


def make_flow_graphs():
    idf_sql_path = []
    json_paths = []
    data_folder_name = ""

    def make_one_graph(case_data: CaseData, json_file_path: Path):
        case_data = CaseData()
        G = make_flow_graph()
        FlowGraphModel.write(
            G,
        )

    pass
