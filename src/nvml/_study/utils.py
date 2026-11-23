# plotting
import matplotlib
import matplotlib.pyplot as plt  # noqa

# data analysis
import numpy as np  # noqa
import polars as pl  # noqa
import polars.selectors as cs  # noqa
import seaborn as sns  # noqa  #noqa
import xarray as xr  # noqa

# jump into ptpython
from nvml import DataPaths as dp  # noqa
from nvml import dn  # noqa

matplotlib.use("module://matplotlib-backend-kitty")


__all__ = [
    # plotting
    "plt",
    "sns",
    "np",
    "pl",
    "cs",
    "xr",
    "dp",
    "dn",
]
