import matplotlib.pyplot as plt  # noqa
import numpy as np
import polars as pl  # noqa
import polars.selectors as cs  # noqa
import seaborn as sns  # noqa

from nvml.df_names import dfc  # noqa


def plot_corr(
    df: pl.DataFrame, drop_qoi: bool = True
):  # this can be reusable for other corr plots, ie the one where well check qoi corr for all wind dirs..
    # possibly drop qoi data
    if drop_qoi:
        df = df.drop(dfc.get_qois())

    corr = df.to_pandas().corr(numeric_only=True)
    mask = np.triu(np.ones_like(corr, dtype=bool))

    f, ax = plt.subplots()

    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    g = sns.heatmap(
        corr,
        mask=mask,
        cmap=cmap,
        center=0,
        square=True,
        linewidths=0.5,
        # cbar_kws={"shrink": 0.5},
        annot=True,
    )
    return g.figure
