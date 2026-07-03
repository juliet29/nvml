import marimo

__generated_with = "0.23.11"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    from nvml.cli.studies.qdim import load_config, get_data_for_graph, get_all_data
    from nvml.cli.config import CONFIGS_DICT
    import altair as alt
    import seaborn.objects as so
    import seaborn as sns

    import polars as pl
    from nvml.constants import DataNames as dn
    from nvml.utils import xr_to_polars

    return dn, get_all_data, mo, pl, so


@app.cell
def _(get_all_data):
    G, qoi, amb = get_all_data()
    return amb, qoi


@app.cell
def _(qoi):
    qoi
    return


@app.cell
def _(amb):
    amb
    return


@app.cell
def _(amb, dn):
    ws = amb[dn.wind_sector]
    ws
    return


@app.cell
def _(amb, dn):
    qoi_coords = {
        dn.wind_sector : (dn.datetime, amb[dn.wind_sector].data),
        dn.wind_dir : (dn.datetime, amb[dn.wind_dir].data),
    }
    return (qoi_coords,)


@app.cell
def _(dn, qoi, qoi_coords):
    qa = qoi.expand_dims([dn.wind_dir, dn.wind_sector]).assign_coords(qoi_coords)
    qa
    return (qa,)


@app.cell
def _(dn, pl, qa):
    df = pl.from_pandas(qa.reset_index(dims_or_levels=[dn.datetime, dn.space_name]).to_dataframe())
    df
    return (df,)


@app.cell
def _(df, dn, so):
    so.Plot(df, dn.wind_sector).add(so.Bars(), so.Hist()).show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Study wind sector wind direction relationship
    """)
    return


@app.cell
def _(df, dn, pl):
    nedf = df.filter(pl.col(dn.wind_sector) == "N")
    nedf
    return (nedf,)


@app.cell
def _(dn, nedf, so):
    so.Plot(nedf, dn.zone_inflow).add(so.Area(), so.Hist(), color=dn.space_name).show()
    return


@app.cell
def _(dn, nedf, so):
    # lets look at the dist of wind directions.. 
    so.Plot(nedf, dn.wind_dir).add(so.Bars(), so.Hist(bins=80)).show()
    return


@app.cell
def _(dn, nedf, pl):
    # a table gives more of the info that is needed.. # TODO: put in utils
    wddf = nedf.select(pl.col(dn.wind_dir).value_counts()).unnest().sort(by="count", descending=True)
    wddf
    return (wddf,)


@app.cell
def _(dn, wddf):
    topwd = wddf.head(3).select(dn.wind_dir)
    topwd
    return (topwd,)


@app.cell
def _(dn, nedf, pl, topwd):
    nedf_wd =  nedf.filter(pl.col(dn.wind_dir).is_in([]))
    nedf_wd = nedf.join(topwd, on=dn.wind_dir)
    return (nedf_wd,)


@app.cell
def _(dn, nedf_wd, so):
    p = so.Plot(nedf_wd, dn.zone_inflow).facet(row=dn.wind_dir).add(so.Area(), so.Hist(), color=dn.space_name).show()
    return


if __name__ == "__main__":
    app.run()
