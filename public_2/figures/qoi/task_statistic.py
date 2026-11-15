from pathlib import Path
from typing import Annotated

import xarray as xr
from pytask import Product

from nvml.paths.data_paths import DataPaths
from nvml.paths.settings import settings
from nvml.qoi_study.space.statistics import Q_Summary_Plots

TEST = Path(".")

FIG_PATH = settings.path_to_figures / "qoi"


def task_space_statistics(
    data_path: Path = DataPaths.reg.aug20v2.qoi_dataArray,
    out_path: Annotated[Path, Product] = FIG_PATH / "statistics.pdf",
):
    qda = xr.open_dataarray(data_path)
    qsp = Q_Summary_Plots(qda)
    fig = qsp.make_plot()
    print(f"save to {out_path}")

    fig.savefig(out_path, format="pdf")
