from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl
import seaborn.objects as so
from matplotlib.figure import Figure


class dfc:
    feature = "feature"
    target = "target"
    case_name = "case_name"
    ww = "ww"
    lw = "lw"
    sd = "sd"


def plot_feature_v_target(df: pl.DataFrame, save_path: Path = Path("")):
    p = so.Plot(df, x=dfc.feature, y=dfc.target).add(so.Dot())
    return p


def plot_bar_compare(df: pl.DataFrame):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    ax1.scatter(df[dfc.ww], df[dfc.target])
    ax1.set_xlabel("ww")

    ax2.scatter(df[dfc.lw], df[dfc.target])
    ax2.set_xlabel("lw")

    ax3.scatter(df[dfc.sd], df[dfc.target])
    ax3.set_xlabel("sd")
    return fig


def plot_bar_compare2(df: pl.DataFrame):
    f = Figure(figsize=(9, 4))
    sfs = f.subfigures(nrows=1, ncols=2)
    # print(sfs)
    # return sfsj
    p1 = so.Plot(df, x=dfc.ww, y=dfc.target).add(so.Dot()).on(sfs[0])  # .plot()

    p2 = so.Plot(df, x=dfc.lw, y=dfc.target).add(so.Dot()).on(sfs[1])  # .plot()
    p1.plot()

    return f

    pass


def plot_target_hist(df: pl.DataFrame):
    p = so.Plot(df, x=dfc.target).add(so.Bars(), so.Hist())
    return p
