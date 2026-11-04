from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl
import seaborn.objects as so

from nvml.features.data_maker import dfc

# class dfc:
#     feature = "feature"
#     target = "target"
#     case_name = "case_name"
#     ww = "ww"
#     lw = "lw"
#     sd = "sd"
#


def plot_feature_v_target(df: pl.DataFrame, save_path: Path = Path("")):
    p = so.Plot(df, x=dfc.coincidence, y=dfc.space_sum_time_mean).add(so.Dot())
    return p


def plot_bar_compare(df: pl.DataFrame):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    ax1.scatter(df[dfc.ww], df[dfc.space_sum_time_mean])
    ax1.set_xlabel("ww")

    ax2.scatter(df[dfc.lw], df[dfc.space_sum_time_mean])
    ax2.set_xlabel("lw")

    ax3.scatter(df[dfc.sd], df[dfc.space_sum_time_mean])
    ax3.set_xlabel("sd")
    return fig


def plot_space_sum_time_mean_hist(df: pl.DataFrame):
    p = so.Plot(df, x=dfc.space_sum_time_mean).add(so.Bars(), so.Hist())
    return p
