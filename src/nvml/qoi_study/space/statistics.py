from dataclasses import dataclass

import matplotlib.pyplot as plt
import xarray as xr
from matplotlib.axes import Axes

from nvml.constants import DataNames as dn
from nvml.qdim.wind import WindDirectionBinNames

ALPHA = 0.5


def style_ax(ax: Axes):
    ax.axline(xy1=(0, 0), slope=1, color="r", lw=2)
    ax.set(alpha=ALPHA)


class QTransforms:
    @classmethod
    def time_median(cls, da: xr.DataArray):
        return da.median(dim=dn.wind_sector)

    @classmethod
    def wind_sector_mean(cls, da: xr.DataArray):
        return da.mean(dim=dn.wind_sector)

    @classmethod
    def drn_select(cls, da: xr.DataArray, drn: WindDirectionBinNames):
        return da.sel({dn.wind_sector: drn})


@dataclass
class Q_Space_Summaries:
    da: xr.DataArray

    def space_mean(self):
        return self.da.mean(dim=dn.space_ix)

    def space_median(self):
        return self.da.median(dim=dn.space_ix)

    def space_std(self):
        return self.da.std(dim=dn.space_ix)

    def space_range(self):
        da = self.da
        diff = da.max(dim=dn.space_ix) - da.min(dim=dn.space_ix)
        return abs(diff)


class Q_Summary_Plots:
    def __init__(self, da: xr.DataArray) -> None:
        self.da = da
        self.qs = Q_Space_Summaries(self.da)

    def loc_plot(self, ax: Axes):
        mean = self.qs.space_mean().pipe(QTransforms.time_median)
        median = self.qs.space_median().pipe(QTransforms.time_median)

        ax.scatter(x=mean, y=median)
        ax.set(xlabel="Mean Q*", ylabel="Median Q*")
        style_ax(ax)
        return ax

    def scale_plot(self, ax: Axes):
        std_ = self.qs.space_std().pipe(QTransforms.time_median)
        range_ = self.qs.space_range().pipe(QTransforms.time_median)

        ax.scatter(x=std_, y=range_)
        ax.set(xlabel="Std Q*", ylabel="Q*_max - Q*_min")
        style_ax(ax)
        return ax

    def make_plot(self):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 4))
        self.loc_plot(ax1)
        self.scale_plot(ax2)
        return fig
