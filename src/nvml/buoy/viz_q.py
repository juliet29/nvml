import re

import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns

from nvml import dn
from nvml.buoy.data import BuoyData, MathNames


def mn(name: str):
    return MathNames.get(name)


def plot_q_dim(bd: BuoyData):
    df2 = bd.def_df()
    fig, (a1, a2, a3) = plt.subplots(1, 3)
    sns.scatterplot(
        data=df2,
        x=df2[dn.U],
        y=df2[dn.q_dim],
        hue=df2[dn.theta],
        ax=a1,
        legend=False,
        rasterized=True,
    )
    a1.set_xlabel(MathNames.get(dn.U))

    sns.scatterplot(
        data=df2,
        x=df2[dn.t_dim],
        y=df2[dn.q_dim],
        hue=df2[dn.theta],
        ax=a2,
        legend=False,
        rasterized=True,
    )
    a2.set_xlabel(MathNames.get(dn.t_dim))

    sns.scatterplot(
        data=df2,
        x=df2[dn.Ri],
        y=df2[dn.q_dim],
        hue=df2[dn.theta],
        ax=a3,
        rasterized=True,
    )

    a3.set_xlabel(MathNames.get(dn.Ri))
    legend = a3.get_legend()
    legend.set_title(MathNames.get(dn.theta))

    for a in (a1, a2, a3):
        a.set_ylabel(MathNames.get(dn.q_dim))

    return fig


def facet_q_ri(bd: BuoyData):
    df = bd.def_df()
    df2 = df.with_columns(
        U_quartile=pl.col(dn.U).qcut(4), td_quartile=pl.col(dn.t_dim).qcut(4)
    )
    uq = "U_quartile"
    tq = "td_quartile"

    f1 = sns.relplot(
        data=df2.sort(by=uq), x=dn.Ri, y=dn.q_dim, hue=dn.theta, col=uq, rasterized=True
    )
    f1.set_axis_labels(x_var=MathNames.get(dn.Ri), y_var=MathNames.get(dn.q_dim))
    for col_val, ax in f1.axes_dict.items():
        clean_interval = re.sub(
            r"-?\d+\.\d+(?:e[+-]?\d+)?",
            lambda m: f"{float(m.group()):.2f}",
            str(col_val),
        )
        ax.set_title(rf"$U \in {clean_interval}$")
    # f1.set_titles(f"{MathNames.get(dn)}")
    pass

    f2 = sns.relplot(
        data=df2.sort(by=tq), x=dn.Ri, y=dn.q_dim, hue=dn.theta, col=tq, rasterized=True
    )
    f2.set_axis_labels(x_var=MathNames.get(dn.Ri), y_var=MathNames.get(dn.q_dim))
    for col_val, ax in f2.axes_dict.items():
        clean_interval = re.sub(
            r"-?\d+\.\d+(?:e[+-]?\d+)?",
            lambda m: f"{float(m.group()):.3f}",
            str(col_val),
        )
        ax.set_title(rf"$T^*_i \in {clean_interval}$")
    # .set_titles(r"$T^*_i)$: {col_name:.2f}")
    return f1.figure, f2.figure


def corr_U_t_dim(bd: BuoyData):
    df = bd.def_df()
    f1 = sns.relplot(data=df, x=dn.U, y=dn.t_dim, hue=dn.theta, rasterized=True)
    f1.set_axis_labels(mn(dn.U), mn(dn.t_dim))
    return f1.figure


def dists(bd: BuoyData):
    df = bd.def_df()
    f1 = sns.displot(df, x=dn.Ri, stat="probability").set_xlabels(mn(dn.Ri))
    f2 = sns.displot(df, x=dn.t_dim, stat="probability").set_xlabels(mn(dn.t_dim))
    f3 = sns.displot(df, x=dn.U, stat="probability").set_xlabels(mn(dn.U))
    f4 = sns.displot(df, x=dn.q_dim, stat="probability").set_xlabels(mn(dn.q_dim))

    return f1.figure, f2.figure, f3.figure, f4.figure
