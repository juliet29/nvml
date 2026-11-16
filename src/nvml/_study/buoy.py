import matplotlib
import matplotlib.pyplot as plt
from cyclopts import App

from nvml._study.config.cstore import CONFIGS_DICT
from nvml.buoy.viz_plan import plot_one
from nvml.buoy.viz_temp import plot_temp_days

matplotlib.use("module://matplotlib-backend-kitty")

by = App("by")

cfg = CONFIGS_DICT[
    "msd_aug20"
]  # TODO: really should have .access, because pretty persisitent; feels like a part of settings..


TEST_CASE = "4969"
TEST_ROOM = "room_2"


@by.command()
def fa():
    plot_one(cfg, TEST_CASE)
    plt.show()


@by.command()
def fb():
    return plot_temp_days(cfg, TEST_CASE, TEST_ROOM)


@by.command()
def fc():
    return plot_temp_days(cfg, TEST_CASE, TEST_ROOM)
