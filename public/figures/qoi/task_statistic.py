from pathlib import Path
from typing import Annotated

import xarray as xr
from pytask import Product
from utils4plots import set_dynamic_params
from utils4plots.save import save_mpl_fig

from nvml._paths import DataPaths
from nvml.qoi_study.space.statistics import Q_Summary_Plots
from nvml.settings import settings

LOC = "qoi"
set_dynamic_params(settings.figure_profile)


def task_space_statistics(
    data_path: Path = DataPaths.reg.aug20v2.qoi_dataArray,
    out_path: Annotated[Path, Product] = settings.fig_path(LOC, "statistics"),
):
    qda = xr.open_dataarray(data_path)
    qsp = Q_Summary_Plots(qda)
    fig = qsp.make_plot()
    save_mpl_fig(fig, out_path)
