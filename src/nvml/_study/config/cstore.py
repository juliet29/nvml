from typing import Literal

import polars as pl

# from nvml._study._root.paths import PATH_TO_MSD_TEMP, PATH_TO_TEMP
from nvml._study.config.case_50 import case_names_50
from nvml._study.config.config import MakeConfig
from nvml._paths import DataPaths
from nvml.settings import settings

# TODO: location of external data should be in .env and settings, not hard coded here!
test_create_config = MakeConfig(
    eplus_model_source=settings.msd_eplus_v1,
    case_names=["100268"],
    data_store=DataPaths.fg.test,
)

jun24config = MakeConfig(
    eplus_model_source=settings.msd_eplus_v1,
    case_names=["100268", "100275", "100306"],
    data_store=DataPaths.fg.jun24_3,
)

case50 = MakeConfig(
    eplus_model_source=settings.msd_eplus_v1,
    case_names=case_names_50,
    data_store=DataPaths.fg.jun30_50,
    # json_file_name=""
)


def get_msd_cases_config():
    path = settings.msd_eplus_v2
    cases_file = path / settings.case_names_file
    case_names = (
        pl.read_csv(cases_file, has_header=False).to_series().cast(pl.String).to_list()
    )
    msd_cases = MakeConfig(
        eplus_model_source=path,
        case_names=case_names,
        data_store=DataPaths.fg.aug20,
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
