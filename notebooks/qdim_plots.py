import marimo

__generated_with = "0.23.11"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo 
    from nvml.cli.studies.qdim import fe
    import seaborn.objects as so
    from nvml.constants import DataNames as dn


    return dn, fe, so


@app.cell
def _(fe):
    df = fe()
    df
    return (df,)


@app.cell
def _(df, dn, so):

    p = (so.Plot(df, x=dn.t_out, color=dn.is_external).pair( y=[dn.zone_dimless_flow, dn.zone_inflow]).facet(dn.wind_sector).add(so.Dots()))
    p.show()
    return


@app.cell
def _(df, dn, so):
    k = (so.Plot(df, x=dn.t_out, color=dn.space_name).pair( y=[dn.zone_dimless_flow, dn.zone_inflow]).facet(dn.wind_sector).add(so.Dots()))
    k.show()
    return


if __name__ == "__main__":
    app.run()
