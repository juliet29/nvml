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
    from nvml.cli.studies.cluster import fd
    ## within module
    return dn, fd, np, plt, xr


@app.cell
def _(fd):
    da = fd()
    da
    return (da,)


@app.cell
def _(da, dn):
    sda = da.dropna(dim=dn.space_ix, how="all")
    sda

    return (sda,)


@app.cell
def _(da, dn):
    da[dn.space_ix]
    return


app._unparsable_cell(
    r"""

    fg = sda.isel({dn.case_name: slice(None, 5)o}).plot.line(
      col=dn.case_name, hue=dn.wind_sector, size=4, aspect=1.2
    )
    fg.fig.set_dpi(400)
    plt.show()
    """,
    name="_"
)


@app.cell
def _(dn, plt, sda):
    sda.sortby(dn.wind_sector).isel({dn.case_name:slice(None, 6)}).plot.line(col=dn.case_name, hue=dn.wind_sector,col_wrap=3 )
    plt.show()
    return


@app.cell
def _(dn, np, sda, xr):
    sda_sorted2 = xr.apply_ufunc(
          lambda x: -np.sort(-x, axis=-1),
          sda,
          input_core_dims=[[dn.space_ix]],
          output_core_dims=[[dn.space_ix]],
          dask="parallelized",
          output_dtypes=[sda.dtype],
      )

    return (sda_sorted2,)


@app.cell
def _(dn, plt, sda_sorted2):
    sda_sorted2.sortby(dn.wind_sector).isel({dn.case_name:slice(None, 6)}).plot.line(col=dn.case_name, hue=dn.wind_sector,col_wrap=3 )
    plt.show()
    return


@app.cell
def _(dn, sda):
    sda.sortby(dn.space_ix)
    return


@app.cell
def _(da, dn):
    da.isel({dn.case_name:0}).dropna(dim=dn.space_ix, how="all")
    return


@app.cell
def _(da, dn):

    da.isel({dn.case_name:1}).dropna(dim=dn.space_ix, how="all")
    return


@app.cell
def _(da, dn):
    da.isel({dn.case_name:48}).dropna(dim=dn.space_ix, how="all")
    return


if __name__ == "__main__":
    app.run()
