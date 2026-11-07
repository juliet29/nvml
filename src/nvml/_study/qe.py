import matplotlib
import matplotlib.pyplot as plt
import xarray as xr
from utils4plots import set_dynamic_params

from nvml._study._root.study_paths import StudyPaths
from nvml.constants import FileNames
from nvml.qoi_study.q_theta import q_theta_plot
from nvml.qoi_study.q_variations import Q_Summary_Plots, QTransforms

matplotlib.use("module://matplotlib-backend-kitty")
from cyclopts import App

qoi_explore_app = App("qe")


def get_data():
    d = xr.open_dataarray(StudyPaths.data.reg.aug20v2 / FileNames.qoi_dataArray)
    return d


def get_temp_data():
    d = xr.open_dataarray(StudyPaths.data.reg.aug20v2 / FileNames.tdim_dataArray)
    return d


@qoi_explore_app.command()
def fa():
    data = get_data()
    return data


@qoi_explore_app.command()
def fb():
    qda = get_data()
    tda = get_temp_data()
    return qda, tda


@qoi_explore_app.command()
def qsummary():
    data = get_data()
    qsp = Q_Summary_Plots(data)
    return qsp


@qoi_explore_app.command()
def qvarplt():
    set_dynamic_params()
    data = get_data().pipe(QTransforms.drn_select, drn="N")
    qsp = Q_Summary_Plots(data)
    qsp.make_plot()
    plt.show()


@qoi_explore_app.command()
def qthetaplt():
    data = get_data()
    return q_theta_plot(data)
