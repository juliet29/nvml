import numpy as np
import pint
import xarray as xr
from metpy.calc import wind_components
from metpy.units import units

SPEED = 1 * units("m/s")


def wind_angles_to_vector(angle: float):
    """angle is in degrees"""
    degrees = angle * units.deg
    wc: tuple[pint.Quantity, pint.Quantity] = wind_components(SPEED, degrees)
    wcm = np.array([wc[0].magnitude, wc[1].magnitude])
    # logger.debug(wcm)
    res = np.round(wcm, decimals=3)

    # logger.debug(res)
    return res


def wind_angle_da_to_vectors(angle: float):
    # fx = lambda x: wind_angles_to_vector(x)
    return xr.apply_ufunc(wind_angles_to_vector, angle)
