import marimo

__generated_with = "0.23.11"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import altair as alt
    import seaborn.objects as so
    import seaborn as sns
    import polars as pl
    import matplotlib.pyplot as plt

    import xarray as xr
    import numpy as np

    ## top level
    from nvml.constants import DataNames as dn
    ## cli
    from nvml.cli.studies.cluster import fda, ff, fc
    ## within module
    from nvml.cluster.spectral import plot_cluster_on_tsne, plot_cluster_on_data

    return dn, fc, fda, ff, plot_cluster_on_data, plot_cluster_on_tsne, plt


@app.cell
def _(fc):
    data = fc()
    data
    return (data,)


@app.cell
def _(fda):
    Y_tsne = fda()
    Y_tsne
    return (Y_tsne,)


@app.cell
def _(ff):
    labels = ff()
    labels
    return (labels,)


@app.cell
def _(data, labels, plot_cluster_on_data, plt):
    plot_cluster_on_data(labels, data)
    plt.show()
    return


@app.cell
def _(Y_tsne, labels, plot_cluster_on_tsne):
    ds = plot_cluster_on_tsne(labels, Y_tsne)
    ds
    return (ds,)


@app.cell
def _(dn, ds):
    ds.plot.scatter(x=dn.c1, y=dn.c2, hue=dn.label)
    return


if __name__ == "__main__":
    app.run()
