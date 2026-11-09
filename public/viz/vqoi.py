from functools import cached_property
from pathlib import Path
from typing import Literal

import matplotlib
import matplotlib.pyplot as plt
import xarray as xr
from cyclopts import App
from dotenv import load_dotenv
from utils4plots import FigSizes, save_mpl_fig, set_dynamic_params

from nvml._study._root.study_paths import StudyPaths
from nvml.constants import DataNames as dn
from nvml.constants import FileNames
from nvml.public_paths import PATH_TO_VIZ
from nvml.qoi_study.space.dist import make_dist_plot
from nvml.qoi_study.space.statistics import Q_Summary_Plots
from nvml.qoi_study.temperature.qt import make_qt_plot, make_qtw_plot

matplotlib.use("module://matplotlib-backend-kitty")

app = App("viz")

FigNames = Literal[
    "space_dist_q",
    "space_dist_t",
    "space_statistics",
    "temperature_qt",
    "temperature_qtw",
]


class FigureMaker:
    def __init__(self, qda_path: Path | None = None, tda_path: Path | None = None):
        default_path = StudyPaths.data.reg.aug20v2
        self.tda_path = (
            tda_path if tda_path else default_path / FileNames.tdim_dataArray
        )
        self.qda_path = qda_path if qda_path else default_path / FileNames.qoi_dataArray

        self.FIG_SAVE_PATH = PATH_TO_VIZ / "vqoi"

    @cached_property
    def qda(self):
        return xr.open_dataarray(self.qda_path)

    @cached_property
    def tda(self):
        return xr.open_dataarray(self.tda_path)

    def get_method(self, name: FigNames):
        return self.__getattribute__(name)

    def temperature_qtw(self):
        fig = make_qtw_plot(self.qda, self.tda)
        return fig

    def temperature_qt(self):
        fig = make_qt_plot(self.qda, self.tda)
        return fig

    def space_dist_q(self):
        FigSizes.set_larger_figsize()
        fig = make_dist_plot(self.qda, dn.zone_dimless_flow)
        return fig

    def space_dist_t(self):
        FigSizes.set_larger_figsize()
        fig = make_dist_plot(self.tda, dn.zone_dimless_temp)
        return fig

    def space_statistics(self):
        qsp = Q_Summary_Plots(self.qda)
        fig = qsp.make_plot()
        return fig


@app.command()
def make(fn: FigNames, save: bool = False, show: bool = True):
    fm = FigureMaker()
    f = fm.get_method(fn)
    fig = f()
    if show:
        plt.show()
    if save:
        p = fm.FIG_SAVE_PATH / fn
        save_mpl_fig(fig, p)


if __name__ == "__main__":
    # colors = CustomColorObjects.colors_list
    # print(colors)
    # cmap = LinearSegmentedColormap.from_list("custom", colors)
    # sns.set_theme(rc={"axes.prop_cycle": cycler(color=colors)})
    # sns.set_palette(CustomColorObjects.cmap.colors)
    load_dotenv()
    set_dynamic_params()
    app()
