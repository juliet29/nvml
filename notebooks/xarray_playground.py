import marimo

__generated_with = "0.23.11"
app = marimo.App()


@app.cell
def _():
    import marimo as mo 
    import xarray as xr
    import numpy as np

    from nvml.constants import DataNames as dn
    from nvml.cli.config import CONFIGS_DICT   

    from nvml.cli.studies.qdim import ff

    from nvml.qdim.io import get_ambient_data_as_ds
    from nvml.qdim.angles import wind_angles_to_vector
    from nvml.qdim.incident import wind_angle_da_to_vectors



    return CONFIGS_DICT, get_ambient_data_as_ds, wind_angle_da_to_vectors


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
    return


if __name__ == "__main__":
    app.run()
