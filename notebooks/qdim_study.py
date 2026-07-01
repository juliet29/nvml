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

    return CONFIGS_DICT, dn, get_data_for_graph, pl


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
    return


@app.cell
def _():
    return


@app.cell
def _(cfg):
    case_name = cfg.get_one_case()
    return (case_name,)


@app.cell
def _(case_name, cfg, get_data_for_graph):
    df = get_data_for_graph(cfg, case_name)

    return (df,)


@app.cell
def _(cfg, dn, get_data_for_graph, pl):
    def create_big_data(case_name):
        df = get_data_for_graph(cfg, case_name)
        return df.group_by(dn.wind_sector).agg(pl.col(dn.zone_dimless_flow).median(), pl.col(dn.incident_factor).min()).with_columns(pl.lit(case_name).alias(dn.case_name))

    return (create_big_data,)


@app.cell
def _(cfg, create_big_data):
    dfs = []
    for i in cfg.case_names:
        try: 
            r = create_big_data(i);
        except Exception as e: 
            print([i, e])
            pass
        dfs.append(r)
        

    return (dfs,)


@app.cell
def _():
    list(range(50))[0:-1:5]
    return


@app.cell
def _(cfg, df2, get_data_for_graph):
    def get_data(stride:int = 5):
        dfs2 = []
        for i in cfg.case_names[0:-1:stride]:
            try: 
                r = get_data_for_graph(i);
            except Exception as e: 
                print([i, e])
                pass
            dfs2.append(r)
        return df2 

    return (get_data,)


@app.cell
def _(get_data):
    dfs2 = get_data()
    return


@app.cell
def _(dfs, pl):
    res = pl.concat(dfs, how="align")
    res
    return (res,)


@app.cell
def _(dn, res):
    res.plot.scatter(x=dn.incident_factor, y=dn.zone_dimless_flow, column=dn.wind_sector)
    return


@app.cell
def _(df, pl):
    df.filter(pl.col("wind_sector")  == "NW")
    return


@app.cell
def _(case_name, df, dn, pl):
    df.group_by(dn.wind_sector).agg(pl.col(dn.zone_dimless_flow).median(), pl.col(dn.incident_factor).min()).with_columns(pl.lit(case_name).alias(dn.case_name))
    return


if __name__ == "__main__":
    app.run()
