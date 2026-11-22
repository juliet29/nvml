from pathlib import Path
from typing import Annotated

import matplotlib.pyplot as plt
from cyclopts import App, Parameter
from utils4plots import save_mpl_fig

from nvml import dp
from nvml.flearn.pca.make_pca import PCAId
from nvml.rprep.collate import (
    RegDataPaths,
)
from nvml.rprep.viz_collate import plot_corr
from nvml.rrun.reg_data_prep import DataSpec, prepare_data
from nvml.settings import settings

fig_feature_corr = App("ffc")
LOC = "feature_corr"

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


@fig_feature_corr.command()
def corr(
    out: Path = settings.fig_path(LOC, "corr"),
    data_spec: Annotated[DataSpec, Parameter(name="*")] = default_data_spec,
):
    # TODO: this is not really qoi validity, more like a feature metric..
    df = prepare_data(*data_spec)

    with plt.rc_context({"figure.figsize": (15, 15)}):
        fig = plot_corr(df)
        save_mpl_fig(fig, out)  # pyright: ignore[reportArgumentType]
