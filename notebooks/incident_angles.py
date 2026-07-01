import marimo

__generated_with = "0.23.11"
app = marimo.App()


@app.cell
def _():
    import marimo as mo 
    import xarray as xr
    import numpy as np
    import matplotlib.pyplot as plt
    import polars as pl

    from nvml.constants import DataNames as dn
    from nvml.utils import xr_to_polars

    from nvml.cli.config import CONFIGS_DICT   
    from nvml.cli.studies.qdim import ff

    from nvml.io import get_ambient_data_as_ds
    from nvml.qdim.angles import wind_angles_to_vector
    from nvml.qdim.incident import wind_angle_da_to_vectors



    return (
        CONFIGS_DICT,
        ff,
        get_ambient_data_as_ds,
        np,
        pl,
        plt,
        wind_angle_da_to_vectors,
    )


@app.cell
def _():
    return


@app.cell
def _(CONFIGS_DICT, get_ambient_data_as_ds):
    cfg = CONFIGS_DICT["jun24"]  
    ads = get_ambient_data_as_ds(cfg.get_one_case_data().sql)   
    return (ads,)


@app.cell
def _(ads):
    wd = ads.wind_direction
    wd
    return (wd,)


@app.cell
def _(wd, wind_angle_da_to_vectors):
    wdv = wind_angle_da_to_vectors(wd)
    wdv
    return (wdv,)


@app.cell
def _(plt, wdv):
    fig, axs = plt.subplots(ncols=2)
    wdv.isel({"xy_vector": 0}).plot.hist(ax=axs[0])
    wdv.isel({"xy_vector": 1}).plot.hist(ax=axs[1])
    return


@app.cell
def _(ff):
    cda = ff()
    cda
    return (cda,)


@app.cell
def _(cda, wdv):
    ia = cda @ wdv.T
    ia
    return (ia,)


@app.cell
def _(ia):
    type(ia.dims)
    return


@app.cell
def _(ia):
    iam = ia.min(dim="edge_num")
    iam
    return (iam,)


@app.cell
def _(iam, pl):
    iam.name="incident_factor"
    pl.from_pandas(iam.to_dataframe().reset_index())
    return


@app.cell
def _():
    import polars as pl

    return (pl,)


@app.cell
def _(iam, np):
    # change incidenct factor to degrees
    id = np.rad2deg(np.arccos(iam))
    id
    return (id,)


@app.cell
def _(id, plt):
    def _():
        fig2, ax = plt.subplots()
        id.plot(hue="space_names", ax=ax)
    _()
    plt.show()
    return


@app.cell
def _(ia, plt):
    fig2, ax = plt.subplots()
    ia.min(dim="edge_num").plot(hue="space_names", ax=ax)
    return


@app.cell
def _(np, plt, wdv):
    def _(np, plt, wdv):
      t = wdv.datetimes.values
      u = wdv.sel(xy_vector="x").values
      v = wdv.sel(xy_vector="y").values

      fig, ax = plt.subplots(figsize=(12, 2.5))
      s = slice(None, None, 200)   # every 12th timestep
      ax.quiver(t[s], np.zeros_like(u[s]), u[s], v[s], angles="uv",
      pivot="mid")

      #ax.quiver(t, np.zeros_like(u), u, v, angles="uv", pivot="mid")
      ax.set_yticks([])
      ax.set_xlabel("datetime")
      return

    fig3 = _(np, plt, wdv)
    plt.show()
    return


if __name__ == "__main__":
    app.run()
