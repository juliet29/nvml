import marimo

__generated_with = "0.23.11"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    from nvml.cli.studies.flow import fc
    import xarray as xr


    return fc, xr


@app.cell
def _(fc):
    q_dims, mix = fc()
    q_dims
    return mix, q_dims


@app.cell
def _(q_dims):
    q = q_dims[0]
    q.drop_vars("space_names").expand_dims("edge")
    return


@app.cell
def _():
    return


@app.cell
def _(q_dims, xr):
    qs = [q.drop_vars("space_names") for q in q_dims]
    da = xr.concat(qs, dim="edge_name")
    da
    return (da,)


@app.cell
def _(mix):
    mix
    return


@app.cell
def _():
    return


@app.cell
def _(mix):
    mix.indexes
    return


@app.cell
def _(da, mix):
    da.assign_coords(mix)
    return


if __name__ == "__main__":
    app.run()
