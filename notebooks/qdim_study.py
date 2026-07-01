import marimo

__generated_with = "0.23.11"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    from nvml.cli.studies.qdim import load_config, get_data_for_graph
    from nvml.cli.config import CONFIGS_DICT
    import altair as alt

    import polars as pl
    from nvml.constants import DataNames as dn

    return CONFIGS_DICT, alt, dn, get_data_for_graph, pl


@app.cell
def _(CONFIGS_DICT):
    CONFIGS_DICT.keys()
    return


@app.cell
def _(CONFIGS_DICT):
    cfg = CONFIGS_DICT["case50"]
    cfg
    return (cfg,)


@app.cell
def _():
    import io
    import sys
    from contextlib import contextmanager

    @contextmanager
    def suppress_stdout():
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            yield
        finally:
            sys.stdout = old_stdout
    # usage
    # with suppress_stdout():
    #     for ii in range(10):
    #         print(ii)
    return (suppress_stdout,)


@app.cell
def _(cfg):
    case_name = cfg.get_one_case()
    return (case_name,)


@app.cell
def _(case_name, cfg, get_data_for_graph):
    df = get_data_for_graph(cfg, case_name)
    return


@app.cell
def _(get_data_for_graph, suppress_stdout):
    def get_data(cfg, stride:int = 5):
        cnt = 0
        dfs = []
        for i in cfg.case_names[0:-1:stride]:
            print(f"{cnt}| Running case {i}")
            cnt+=1
            try: 
                with suppress_stdout():
                    r = get_data_for_graph(cfg, i);
            except Exception as e: 
                print([i, e])
                continue
            dfs.append(r)
        return dfs 

    return (get_data,)


@app.cell
def _(cfg, get_data):
    dfs2 = get_data(cfg)
    return (dfs2,)


@app.cell
def _(dfs2, dn, pl):
    tdf = dfs2[4].filter(pl.col(dn.wind_sector) == "E")
    tdf
    return (tdf,)


@app.cell
def _(alt, dn, tdf):
    tdf.plot.scatter(x=alt.X(dn.wind_dir).scale(zero=False), y=dn.zone_dimless_flow, color=alt.Color(dn.incident_factor, type="ordinal").scale(scheme="lightgreyred"))
    return


@app.cell
def _(dn, tdf):
    tdf.plot.bar(x=dn.wind_dir, y="count()")
    return


app._unparsable_cell(
    r"""
    tdff2 = tdf.filter(pl.col(dn.wind_dir) == 90).group_by_dynamic(dn.datetime, every="48h").
    tdff2
    """,
    name="_"
)


@app.cell
def _(dn, pl, tdf):
    tdff = tdf.filter(pl.col(dn.wind_dir) == 90).group_by(dn.datetime).head(5)
    tdff 
    #.plot.bar(x=dn.zone_dimless_flow, y="count()", color=dn.space_name)
    return (tdff,)


app._unparsable_cell(
    r"""

    alt.Chart(tdff).transform_density(
        density=dn.zone_dimless_flow,
        groupby=['Species'],
        extent= [2500, 6500],
        counts = True,
        steps=200
    ).mark_area().encode(
        alt.X('value:Q').title('Body Mass (g)'),
        alt.Y('density:Q', stack='zero'),
        alt.Color('Species:N')
    """,
    name="_"
)


@app.cell
def _(dn, tdff):

    tdff.plot.bar(x=dn.zone_dimless_flow, y="count()", color=dn.space_name, column=dn.wind_speed)
    return


@app.cell
def _(dn, tdff):
    tdff.plot.bar(x=dn.zone_dimless_flow, y="count()", color=dn.space_name, column=dn.datetime)
    return


@app.cell
def _(dn, pl, tdf):
    tdf.filter(pl.col(dn.wind_dir) == 90).select(pl.col(dn.datetime).unique())
    return


@app.cell
def _(dn, pl, tdf):
    tdf.select(pl.col(dn.wind_dir).value_counts())
    return


@app.cell
def _():
    # spatial dist for given wind directions
    return


if __name__ == "__main__":
    app.run()
