from pathlib import Path

from nvml._study.config.config import MakeConfig
from nvml.regression.models.collate import prepare_qoi_df


def plot_qoi_outliers(path: Path, cfg: MakeConfig):
    qoi_df = prepare_qoi_df(path)

    # read in qois
    # for each var, plot graphs of the 5 larges
    # and plot plans of the 5 largest
    # likely will want to re-do the kernel with these taken out..
    # later, watnt to look at what happens when remove these outliers..
    pass
