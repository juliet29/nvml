import joblib
import matplotlib
from cyclopts import App
from loguru import logger
from ptpython.repl import embed

# from nvml import DataPaths as dp
# from nvml import dn  # noqa
from nvml._study.config.cstore import CONFIGS_DICT
from nvml._study.utils import *
from nvml.rprep.collate import DataSpec, collate_data_from_files
from nvml.rprep.df_names import dfc  # noqa
from nvml.rrun.defn import simple
from nvml.rrun.make_summary import summary_save
from nvml.rrun.run import gamma_model, multi_model_save

matplotlib.use("module://matplotlib-backend-kitty")
r1_app = App("r1")


cfg = CONFIGS_DICT["msd_aug20"]
REG = dp.reg.v3
PCA = dp.kern.v2.pca_h5
QOI = REG.flow_qoi_dataset
data_paths = (REG.const_features, REG.wdep_features, PCA, QOI)

dsp = DataSpec(*data_paths, "N", "wl", "int_ext", 5)

TEMP = dp.temp.base
TEST_NAME = "simple_test"


@r1_app.command()
def ff():

    df = collate_data_from_files(dsp)
    defns, name = simple()
    logger.debug(defns)
    models = [gamma_model(i, df) for i in defns]
    multi_model_save(TEST_NAME, defns, models, TEMP)


@r1_app.command()
def ff2():
    df = collate_data_from_files(dsp)
    defns, name = simple()
    logger.debug(defns)
    p = summary_save(df, TEST_NAME, defns, TEMP)
    return p


@r1_app.command()
def fg():
    p = TEMP / f"{TEST_NAME}.joblib"
    res = joblib.load(p)

    embed(globals(), locals(), vi_mode=True)
