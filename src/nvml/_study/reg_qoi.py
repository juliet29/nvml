import matplotlib
import xarray as xr
from cyclopts import App

from nvml.constants import DataNames as dn
from nvml.constants import FileNames
from nvml.features.dataset_maker import make_dataset, make_tdim_dataset

matplotlib.use("module://matplotlib-backend-kitty")

from nvml._study._root.study_paths import StudyPaths
from nvml._study.config.cstore import CONFIGS_DICT

reg_qoi_app = App("rq")


# @reg_qoi_app.command()
# def try_one():
#     cfg = CONFIGS_DICT["msd_aug20"]
#     direction = "N"
#     ambient_ds = get_ambient_ds_from_cfg(cfg)
#     dm = DataMaker(ambient_ds, direction, cfg, 100)
#     return dm.make_table_entry()
#


@reg_qoi_app.command()
def mk_data(true_run: bool = False, save: bool = False, njobs: int = 3):
    # have 24 cores? so 18 is not a bad choice
    cfg = CONFIGS_DICT["msd_aug20"]
    # direction = "N"
    save_path = StudyPaths.data.reg.aug20v2

    return make_dataset(cfg, save_path, true_run, save, njobs)


@reg_qoi_app.command()
def mk_tdim(true_run: bool = False, save: bool = False, njobs: int = 3):
    # have 24 cores? so 18 is not a bad choice
    cfg = CONFIGS_DICT["msd_aug20"]
    # direction = "N"
    save_path = StudyPaths.data.reg.aug20v2

    return make_tdim_dataset(cfg, save_path, true_run, save, njobs)


@reg_qoi_app.command()
def mk_3h():
    save_path = StudyPaths.data.reg.aug20v2
    qda = (
        xr.open_dataarray(save_path / FileNames.qoi_dataArray)
        .resample({dn.datetime: "3h"})
        .mean()
    )

    tda = (
        xr.open_dataarray(save_path / FileNames.tdim_dataArray)
        .resample({dn.datetime: "3h"})
        .mean()
    )
    qda.to_netcdf(save_path / FileNames.qoi_3h_dataArray)
    tda.to_netcdf(save_path / FileNames.tdim_3h_dataArray)
