from nvml._study.qe import fa, fb  # noqa
import polars as pl  # noqa
import pandas as pd  # noqa
from nvml.constants import DataNames as dn  # noqa
import matplotlib.pyplot as plt  # noqa
import xarray as xr  # noqa
from nvml.qoi_study.temperature.scatter_zone import add_wind_sector_codes  # noqa

import matplotlib

matplotlib.use("module://matplotlib-backend-kitty")

qda, tda, tza = fb()
q2 = add_wind_sector_codes(qda)
# ds = q_tdim_setup(qda, tda)
