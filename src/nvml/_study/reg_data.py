import matplotlib
from cyclopts import App

matplotlib.use("module://matplotlib-backend-kitty")

from nvml._study._root.study_paths import StudyPaths
from nvml._study.config.cstore import CONFIGS_DICT
from nvml.features.data_maker import save_data

reg_data_app = App("rd")


@reg_data_app.command()
def save():
    cfg = CONFIGS_DICT["msd_aug20"]
    direction = "N"
    save_path = StudyPaths.data.anova.aug20v2

    # "are you sure you want to overwerite this path?"
    return save_data(cfg, direction, save_path)
