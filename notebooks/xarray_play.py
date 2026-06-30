import marimo

__generated_with = "0.23.11"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import xarray as xr
    import numpy as np
    import pandas as pd

    return mo, np, pd, xr


@app.cell
def _(xr):
    help(xr.tutorial)
    return


@app.cell
def _(xr):
    ds = xr.tutorial.open_dataset("air_temperature")
    ds
    return (ds,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Selecting Data
    """)
    return


@app.cell
def _(ds):
    da = ds.air
    return (da,)


@app.cell
def _(da, ds):
    da.sel(time=ds.air.time)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Piping functions / functional analysis
    """)
    return


@app.cell
def _(da):
    fx = lambda x: x*2
    da.pipe(fx)
    return


@app.cell
def _():
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## MultiIndex
    """)
    return


@app.cell
def _(np, xr):
    time = np.arange(20)
    eix = [("a", "b"), ("b", "c"), ("c", "d")]
    coords = {
        "time": time, 
        #"eix": eix
    }

    da2 = xr.DataArray(np.ones(shape=(len(time), len(eix))), coords=coords, dims = ["time", "edge"])
    da2
    return da2, eix


@app.cell
def _(eix, pd):
    midx = pd.MultiIndex.from_tuples(eix, names=("u", "v"))
    midx
    return (midx,)


@app.cell
def _(midx, xr):
    midx_coords = xr.Coordinates.from_pandas_multiindex(midx, dim="edge")
    midx_coords
    return (midx_coords,)


@app.cell
def _(da2, midx_coords):
    da2c = da2.assign_coords(midx_coords)
    da2c
    return (da2c,)


@app.cell
def _(da2c):
    da2c.sel(edge=("a", "b"))
    return


@app.cell
def _(da2c):
    # won't work
    da2c.sel(edge=("b", "d"))
    return


if __name__ == "__main__":
    app.run()
