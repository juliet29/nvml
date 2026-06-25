from pathlib import Path
from typing import NamedTuple

from plyze import CaseData


class Paths:
    data_store = Path()
    pass


class Config(NamedTuple):
    eplus_model_source: Path
    data_store: Path
    json_file_name: str
    data_folder_name: str
    case_names: list[str]
    idf_file_name: str
    sql_file_name: str

    def create_case_datas(self):
        def make(case_name):
            idf_path = self.eplus_model_source / case_name / self.idf_file_name
            sql_path = self.eplus_model_source / case_name / self.sql_file_name
            return CaseData(idf_path, sql_path)

        return [make(i) for i in self.case_names]
