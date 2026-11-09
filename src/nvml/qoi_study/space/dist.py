import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import xarray as xr
from matplotlib.axes import Axes
from utils4plots import CustomColorObjects

from nvml.constants import DataNames as dn


# TODO: may be used for temperature data!
def make_dist_plot(da: xr.DataArray, qoi: str):
    def make_one_data(ix: int):
        da1 = da.isel({dn.case_name: ix})
        case_name: str = da1[dn.case_name].values.tolist()

        df = da1.to_pandas().T
        assert isinstance(df, pd.DataFrame)
        df2 = (
            df.reset_index(drop=True)
            .melt(
                # id_vars=dn.wind_sector,
                var_name=dn.space_ix,
                value_name=qoi,
            )
            .dropna(subset=[qoi])
        )

        return df2, case_name

    def make_one_plot(df: pd.DataFrame, ax: Axes, case_name: str):
        res = sns.kdeplot(
            df,
            x=qoi,
            hue=dn.space_ix,
            fill=True,
            # multiple="stack",
            ax=ax,
            clip=(0, 0.4),
            palette=CustomColorObjects.seg_cmap,  # pyright: ignore[reportAttributeAccessIssue]
        )
        ax.set_title(case_name)
        return res

    fig, ax = plt.subplots(nrows=1, ncols=3)
    for i in range(3):
        df, case_name = make_one_data(i)
        make_one_plot(df, ax[i], case_name)

    return fig
