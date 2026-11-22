from pathlib import Path
from typing import Annotated

from cyclopts import App, Parameter

from nvml._study.config.cstore import CONFIGS_DICT
from nvml._study.f2 import dp
from nvml.flearn.pca.make_pca import PCAId
from nvml.rprep.collate import (
    RegDataPaths,
)
from nvml.rrun.reg_data_prep import DataSpec, prepare_data
from nvml.rrun.reg_interfaces import (
    ExperimentConfig,
)
from nvml.rrun.reg_run import run_experiment

glm = App("glm")

cfg = CONFIGS_DICT["msd_aug20"]
REG_SRC = dp.reg.v3
KERN_SRC = dp.kern.v2

pca_id = PCAId("wl", "int_ext")
default_data_spec = DataSpec(
    flow_qoi=REG_SRC.flow_qoi_dataset,
    pca_path=KERN_SRC.pca_h5,
    rdp=RegDataPaths(REG_SRC.const_features, REG_SRC.wdep_features),
    n=5,
    drn="N",
    pcaid=pca_id,
)

TEST_DIR = Path("../../../")
GLM_CONFIG_PATH = Path(
    "/home/groups/risheej/jnwagwu/fpopt/nvml/public/configs/reg_models/_01.yaml"
)


@glm.command()
def make(
    config_path: Path = GLM_CONFIG_PATH,
    data_spec: Annotated[DataSpec, Parameter(name="*")] = default_data_spec,
    out: Path = dp.temp.base,
):
    cfg = ExperimentConfig.from_yaml(config_path)
    df = prepare_data(*data_spec)
    results = run_experiment(cfg, df, data_spec)
    results.save(out)
