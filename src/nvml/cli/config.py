from pathlib import Path
from typing import NamedTuple

from plyze import CaseData

from nvml.cli.paths import PATH_TO_STORAGE, PATH_TO_TEMP


class MakeConfig(NamedTuple):
    eplus_model_source: Path
    case_names: list[str]
    data_store: Path
    json_file_name: str = "out.json"
    data_folder_name: str = "data"
    idf_file_name: str = "out.idf"
    sql_file_name: str = "eplusout.sql"
    cardinal_expansion_factor = 1.1

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


test_create_config = MakeConfig(
    eplus_model_source=PATH_TO_STORAGE / "msherlock_v1/data",
    case_names=["100268"],
    data_store=PATH_TO_TEMP / "test_create",
)

jun24config = MakeConfig(
    eplus_model_source=PATH_TO_STORAGE / "msherlock_v1/data",
    case_names=["100268", "100275", "100306"],
    data_store=PATH_TO_TEMP / "jun24",
)

case50 = MakeConfig(
    eplus_model_source=PATH_TO_STORAGE / "msherlock_v1/data",
    case_names=[
        "100268",
        "100270",
        "100275",
        "100278",
        "100306",
        "100310",
        "100347",
        "100359",
        "100397",
        "100403",
        "100451",
        "100489",
        "100640",
        "100642",
        "100712",
        "100759",
        "100785",
        "100788",
        "100798",
        "100916",
        "100921",
        "101043",
        "101049",
        "101331",
        "101364",
        "10156",
        "101833",
        "101833",
        "101935",
        "102751",
        "102757",
        "10317",
        "10335",
        "103450",
        "104498",
        "104689",
        "104739",
        "104786",
        "105741",
        "105957",
        "10625",
        "106340",
        "106430",
        "106473",
        "106493",
        "106521",
        "10656",
        "106584",
        "106664",
        "106675",
        "106718",
    ],
    data_store=PATH_TO_TEMP / "jun30",
    # json_file_name=""
)


CONFIGS_DICT = {
    "jun24": jun24config,
    "test_create": test_create_config,
    "case50": case50,
}
