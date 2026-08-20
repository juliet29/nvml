from typing import Literal

from utils4plans.io import read_json

from nvml.cli.config.case_50 import case_names_50
from nvml.cli.config.config import MakeConfig
from nvml.cli.paths import PATH_TO_MSD_TEMP, PATH_TO_STORAGE, PATH_TO_TEMP

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
    case_names=case_names_50,
    data_store=PATH_TO_TEMP / "jun30",
    # json_file_name=""
)


def get_msd_cases_config():
    path = PATH_TO_MSD_TEMP / "test_msd"
    case_json = read_json(path / "batch_reports/merged_quick.json")
    cases = case_json["cases_by_category"]["SUCCEEDED"]
    str_cases = [str(i) for i in cases]
    msd_cases = MakeConfig(
        eplus_model_source=path / "model",
        case_names=str_cases,
        data_store=PATH_TO_TEMP / "msd_aug20",
        sql_file_name="results/eplusout.sql",
    )

    return msd_cases


config_names = Literal["jun24", "case50", "msd_aug20", "test_create"]
CONFIGS_DICT: dict[config_names, MakeConfig] = {
    "jun24": jun24config,
    "test_create": test_create_config,
    "case50": case50,
    "msd_aug20": get_msd_cases_config(),
}
