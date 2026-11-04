import matplotlib
import matplotlib.pyplot as plt
import xarray as xr

from nvml._study._root.study_paths import StudyPaths
from nvml._study.reg_arch import get_data, get_data_and_splits
from nvml.constants import FileNames
from nvml.features.visuals import plot_bar_compare, plot_space_sum_time_mean_hist

matplotlib.use("module://matplotlib-backend-kitty")
from cyclopts import App

qoi_explore_app = App("qe")


def get_data():
    d = xr.open_dataarray(StudyPaths.data.reg.aug20v2 / FileNames.qoi_dataArray)

    return d


@qoi_explore_app.command()
def fa():
    data = get_data_and_splits()
    return data


@reg_app.command()
def fcb():
    data = get_data_and_splits()
    p = plot_space_sum_time_mean_hist(data)
    p.show()

    if None:
        plot_bar_compare(data)
    plt.show()


# @reg_app.command()
# def ff():
#     df = get_data().to_pandas()
#     ic(df)
#     coded_data = dmatrix(
#         f"0+ C({dfc.feature}, Treatment(0))", df, return_type="dataframe"
#     )
#     print(coded_data)
#
#     # patsy model for the design matrix
#     pass
#
#
# @reg_app.command()
# def fe():
#     # split model
#     df = get_data_and_splits()  # .to_pandas()
#     logger.info("data is ready...")
#     study_many_formulas(df)
