import marimo

__generated_with = "0.23.11"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    from nvml.cli.studies.flow import fc, fb
    from nvml.utils import xr_to_polars
    import xarray as xr
    import polars as pl
    import altair as alt


    return alt, fb, fc, pl, xr_to_polars


@app.cell
def _(fb):
    paths = fb()
    paths
    return (paths,)


@app.cell
def _(fc):
    qda = fc()
    qda
    return (qda,)


@app.cell
def _(qda):
    qda.edge_name.u
    return


@app.cell
def _(paths, qda):
    pg = paths[0]
    qpg = pg.get_qdim_data(qda)
    #qpg.stack("edge_name")
    qpg.plot.line(x="datetimes")
    return pg, qpg


@app.cell
def _(qpg):
    qpg
    return


@app.cell
def _(qpg):
    # key! just rest the index of edge name, and the info is still there!
    resampled = qpg.T.reset_index("edge_name").resample(datetimes="96h").first()
    resampled.name = "flow"
    resampled
    return (resampled,)


@app.cell
def _(pl, resampled, xr_to_polars):
    df2 = xr_to_polars(resampled).with_columns(pl.col("edge_name").cast(pl.String).cast(pl.Categorical))
    df2
    return (df2,)


@app.cell
def _(pg):
    pg.edge_list
    return


@app.cell
def _(alt, df2):
    df2.plot.line(x="edge_name", y = "flow").encode(color=alt.Color("datetimes", type="ordinal").scale(scheme="lightgreyred")
    ).properties(width=500)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
