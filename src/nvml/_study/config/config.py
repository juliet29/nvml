from pathlib import Path
from typing import NamedTuple

from plyze import CaseData


class MakeConfig(NamedTuple):
    eplus_model_source: Path
    case_names: list[str]
    data_store: Path
    json_file_name: str = "out.json"
    data_folder_name: str = "data"
    idf_file_name: str = "out.idf"
    sql_file_name: str = "eplusout.sql"
    cardinal_expansion_factor = 1.1

    def __len__(self):
        return len(self.case_names)

    def make_case_data(self, case_name):
        idf_path = self.eplus_model_source / case_name / self.idf_file_name
        sql_path = self.eplus_model_source / case_name / self.sql_file_name
        return CaseData(idf_path, sql_path)

    def make_graph_path(self, case_name: str):
        # return self.data_store / case_name / self.json_file_name
        return self.data_store / case_name / self.data_folder_name / self.json_file_name

    # used in studies
    def get_one_case(self, ix: int = 0):
        return self.case_names[ix]

    def get_one_case_data(self, ix: int = 0):
        return self.make_case_data(self.case_names[ix])

    def get_one_graph_path(self, ix: int = 0):
        return self.make_graph_path(self.case_names[ix])

    # TODO: define some kind of iterable over this..
