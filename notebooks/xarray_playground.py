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

    from nvml.qdim.io import get_ambient_data_as_ds
    from nvml.qdim.angles import wind_angles_to_vector


    return CONFIGS_DICT, get_ambient_data_as_ds, mo, np, xr


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
def _():
    return


@app.cell
def _(np, xr):
    def expand(val: np.array):
        # we are being passed the entire array..
        # change it as like, then stack what's to return 
        # axis=-1 means inner most
        return np.stack([val, np.zeros_like(val) ], axis=-1) 

    def premult2(val): return val*2

    def mult2(val: xr.DataArray):
        #fx =  lambda val: expand(val)
        return xr.apply_ufunc(expand, val,  output_core_dims=[["wind_vector_coord"]])

    return (mult2,)


app._unparsable_cell(
    r"""
    def vectorize_wind(arr: np.array):
        print(f"recieved arr of {type(arr)}, shape {arr.shape}")
        res =  wind_angles_to_vector(arr).
        print(f"result is of type {type(res)}, with shape {res.shape}")
        return res
    """,
    name="_"
)


@app.cell
def _(vectorize_wind, xr):
    def calculate_wind_vector_for_da(da: xr.DataArray):
        #fx = lambda x: wind_angles_to_vector(x)
        return xr.apply_ufunc(vectorize_wind, da, input_core_dims=[["datetimes"]], output_core_dims=[["datetimes","xy"]])

    return (calculate_wind_vector_for_da,)


@app.cell
def _(calculate_wind_vector_for_da, wd):
    calculate_wind_vector_for_da(wd)
    return


@app.cell
def _():
    wvc = {"wind_vector_coord": ["x", "y"]}
    wvc
    return


@app.cell
def _(mult2, wd):
    mult2(wd)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Xarray Tutorial
    """)
    return


@app.cell
def _(xr):
    air_da =   xr.tutorial.load_dataset("air_temperature")
    air_da
    return (air_da,)


@app.cell
def _(air_da):
    air = (
      air_da
        .air.sortby("lat")  # np.interp needs coordinate in ascending order
        .isel(time=-0, lon=0)  # choose a 1D subset
    )
    air
    return (air,)


@app.cell
def _(air):
    air.plot.line("b-^")
    return


@app.cell
def _(air, np):
    newlat = np.linspace(15, 75, 100)
    np.interp(newlat, air.lat.data, air.data)
    return (newlat,)


@app.cell
def _(air, newlat, np, xr):
    r1 = xr.apply_ufunc(np.interp, newlat, air.lat, air, input_core_dims=[["lat_interp"], ["lat"], ["lat"]], output_core_dims=[["lat_interp"]])
    r1.plot.line("b-^")
    return (r1,)


@app.cell
def _(r1):
    r1
    return


if __name__ == "__main__":
    app.run()
