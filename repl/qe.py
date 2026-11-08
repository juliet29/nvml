from nvml._study.qe import fa, fb  # noqa
from nvml.constants import DataNames as dn  # noqa
import matplotlib.pyplot as plt  # noqa
import xarray as xr  # noqa
from nvml.qoi_study.q_richardson import q_tdim_setup, q_tdim_plot2  # noqa

import matplotlib

matplotlib.use("module://matplotlib-backend-kitty")

qda, tda = fb()
ds = q_tdim_setup(qda, tda)
