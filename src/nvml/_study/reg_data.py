import matplotlib
from cyclopts import App

from nvml._study.config.config import MakeConfig
from nvml.io import get_ambient_data_as_ds
from nvml.qdim.wind import add_wind_sector_coord

matplotlib.use("module://matplotlib-backend-kitty")

from nvml._study._root.study_paths import StudyPaths
from nvml._study.config.cstore import CONFIGS_DICT
from nvml.features.data_maker import DataMaker, save_data

reg_data_app = App("rd")


def get_ambient_ds(ix: int, cfg: MakeConfig):
    res = get_ambient_data_as_ds(cfg.get_one_case_data(ix).sql)
    res = add_wind_sector_coord(res)
    return res


@reg_data_app.command()
def try_one():
    cfg = CONFIGS_DICT["msd_aug20"]
    direction = "N"
    ambient_ds = get_ambient_ds(0, cfg)
    dm = DataMaker(ambient_ds, direction, cfg, 100)
    return dm.make_table_entry()


@reg_data_app.command()
def mk_data(true_run: bool = False, save: bool = False, njobs: int = 3):
    # have 24 cores? so 18 is not a bad choice
    cfg = CONFIGS_DICT["msd_aug20"]
    direction = "N"
    save_path = StudyPaths.data.anova.aug20v2

    return save_data(cfg, direction, save_path, true_run, save, njobs)
