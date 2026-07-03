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

    ## top level
    from nvml.constants import DataNames as dn
    ## cli
    from nvml.cli.studies.qdim import get_all_data
    ## within module

    return dn, get_all_data


@app.cell
def _(get_all_data):
    G, qoi, amb = get_all_data()
    return G, qoi


@app.cell
def _(G):
    G
    return


@app.cell
def _(dn, qoi):
    q_dim = qoi[dn.zone_dimless_flow]
    #q_dim.sortby(q_dim)
    q_dim
    return (q_dim,)


@app.cell
def _(dn, q_dim):
    q_dim.expand_dims(dim=["space_ix"]).assign_coords({"space_ix": (dn.space_name, range(len(q_dim.space_names.data)))})
    return


@app.cell
def _(dn, q_dim):
    q_dim[dn.space_name].size
    return


@app.cell
def _(dn, q_dim):
    q_dim.assign_coords(space_ix=(dn.space_name,
      range(q_dim.sizes[dn.space_name]))).swap_dims({dn.space_name: "space_ix"}).drop_vars(dn.space_name)

    return


if __name__ == "__main__":
    app.run()
