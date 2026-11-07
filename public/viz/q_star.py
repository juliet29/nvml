import os
from pathlib import Path
from typing import Callable, Literal

import pyprojroot
import xarray as xr
from dotenv import load_dotenv
from matplotlib.figure import Figure
from utils4plots import save_mpl_fig, set_dynamic_params

from nvml._study._root.study_paths import StudyPaths
from nvml.constants import FileNames
from nvml.qoi_study.q_variations import Q_Summary_Plots, QTransforms

BASE_PATH = pyprojroot.find_root(pyprojroot.has_dir(".git"))

load_dotenv(Path(BASE_PATH) / ".env")

storage_path = os.getenv("PATH_TO_TEMP")
assert storage_path
PATH_TO_TEMP = Path(storage_path)
FIG_SAVE_PATH = PATH_TO_TEMP / "viz" / "q_star"


def get_data():
    d = xr.open_dataarray(StudyPaths.data.reg.aug20v2 / FileNames.qoi_dataArray)
    return d


def get_temp_data():
    d = xr.open_dataarray(StudyPaths.data.reg.aug20v2 / FileNames.tdim_dataArray)
    return d


def qvarplt():
    set_dynamic_params()
    data = get_data().pipe(QTransforms.drn_select, drn="N")
    qsp = Q_Summary_Plots(data)
    fig = qsp.make_plot()
    return fig


FigNames = Literal["spatial_summary"]
FigDict: dict[FigNames, Callable[[], Figure]] = {
    "spatial_summary": qvarplt,
}


def save_figure(fig_name: FigNames):
    f = FigDict[fig_name]
    fig = f()
    p = FIG_SAVE_PATH / fig_name
    save_mpl_fig(fig, p)

    pass


if __name__ == "__main__":
    save_figure("spatial_summary")
