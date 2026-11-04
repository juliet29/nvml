import matplotlib
import xarray as xr

from nvml._study._root.study_paths import StudyPaths
from nvml.constants import FileNames

matplotlib.use("module://matplotlib-backend-kitty")
from cyclopts import App

qoi_explore_app = App("qe")


def get_data():
    d = xr.open_dataarray(StudyPaths.data.reg.aug20v2 / FileNames.qoi_dataArray)

    return d


@qoi_explore_app.command()
def fa():
    data = get_data()
    return data
