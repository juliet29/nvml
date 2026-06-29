import marimo

__generated_with = "0.23.11"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import xarray as xr

    return mo, xr


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


if __name__ == "__main__":
    app.run()
