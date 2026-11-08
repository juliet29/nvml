import os
from pathlib import Path
from typing import Callable, Literal

import pyprojroot
import xarray as xr
from cyclopts import App
from dotenv import load_dotenv
from matplotlib.figure import Figure
from utils4plots import save_mpl_fig, set_dynamic_params

from nvml._study._root.study_paths import StudyPaths
from nvml.constants import FileNames
from nvml.qoi_study.q_richardson import q_ri_plot, q_tdim_plot_rooms, q_tdim_wind
from nvml.qoi_study.q_theta import q_theta_plot
from nvml.qoi_study.q_variations import Q_Summary_Plots

BASE_PATH = pyprojroot.find_root(pyprojroot.has_dir(".git"))

load_dotenv(Path(BASE_PATH) / ".env")

storage_path = os.getenv("PATH_TO_TEMP")
assert storage_path
PATH_TO_TEMP = Path(storage_path)
FIG_SAVE_PATH = PATH_TO_TEMP / "viz" / "q_star"

app = App("viz")


@app.command()
def get_data():
    d = xr.open_dataarray(StudyPaths.data.reg.aug20v2 / FileNames.qoi_dataArray)
    return d


def get_temp_data():
    d = xr.open_dataarray(StudyPaths.data.reg.aug20v2 / FileNames.tdim_dataArray)
    return d


def qvarplt():
    data = get_data()  # .pipe(QTransforms.drn_select, drn="N")
    qsp = Q_Summary_Plots(data)
    fig = qsp.make_plot()
    return fig


def qtheta():
    data = get_data()
    fig = q_theta_plot(data)
    return fig


def qtdim():
    qda = get_data()
    tda = get_temp_data()
    fig = q_tdim_wind(qda, tda)
    return fig


def qri():
    qda = get_data()
    tda = get_temp_data()
    fig = q_ri_plot(qda, tda)
    return fig


def qtdimr():
    qda = get_data()
    tda = get_temp_data()
    fig = q_tdim_plot_rooms(qda, tda)
    return fig


FigNames = Literal["q_tdim_r", "space_summary", "q_theta", "q_tdim", "q_ri"]
FigDict: dict[FigNames, Callable[[], Figure]] = {
    "space_summary": qvarplt,
    "q_theta": qtheta,
    "q_tdim": qtdim,
    "q_ri": qri,
    "q_tdim_r": qtdimr,
}


@app.command()
def save(fn: FigNames):
    f = FigDict[fn]
    fig = f()
    p = FIG_SAVE_PATH / fn
    save_mpl_fig(fig, p)

    pass


if __name__ == "__main__":
    set_dynamic_params()
    app()
