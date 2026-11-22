import numpy as np
import polars as pl
import seaborn as sns
import xarray as xr
from loguru import logger
from polars import selectors as cs

from nvml import dn
from nvml.settings import settings


def get_random_plans(da: xr.DataArray | xr.Dataset, n_samples: int = 4):
    rng = np.random.default_rng(seed=settings.random_state)
    idx = rng.choice(da.sizes[dn.case_name], size=n_samples, replace=False)
    return da.isel({dn.case_name: idx})


def prep_df(qoi: str, flow_qoi: xr.Dataset, ambient_ds: xr.Dataset, n_plans: int):
    fq = flow_qoi.drop_vars(dn.wind_sector).pipe(get_random_plans, n_samples=n_plans)[
        qoi
    ]
    ds = ambient_ds.drop_vars(dn.wind_sector)
    ds[qoi] = fq

    df = ds.to_dataframe()
    return df


def plot_ambient_dep_scatter_per_qoi(
    qoi: str,
    flow_qoi: xr.Dataset,
    ambient_ds: xr.Dataset,
    drn_range: tuple = (180, 360),
    n_plans: int = 4,
):

    df = prep_df(qoi, flow_qoi, ambient_ds, n_plans)
    # need to add theta, U, and T to flow_qoi dataset
    # think it has datetime

    # select 10 random plans

    def make(amb_var):
        g = sns.relplot(
            df,
            x=amb_var,
            y=qoi,
            hue=dn.case_name,
            col=dn.case_name,
            kind="scatter",
            legend=False,
            rasterize=True,
        )

        if amb_var == dn.wind_dir:
            g.set(xlim=drn_range)

        # g.figure.suptitle(amb_var)
        return g

    amb_vars = [dn.wind_dir, dn.wind_speed, dn.t_out]
    figs = [make(i).figure for i in amb_vars]
    return figs, amb_vars
    # return make(qoi).figure


def plot_ambient_dep_box_per_qoi(
    qoi: str,
    flow_qoi: xr.Dataset,
    ambient_ds: xr.Dataset,
    drn_range: tuple = (180, 360),
    n_plans: int = 50,
):
    # TODO: Rotate axes or make the plot bigger

    logger.info(f"Making plot for {qoi}")
    pdf = prep_df(qoi, flow_qoi, ambient_ds, n_plans)
    df = (
        pl.from_pandas(pdf, include_index=True)
        .filter(pl.col(dn.wind_dir) > drn_range[0])
        .filter(pl.col(dn.wind_dir) < drn_range[1])
    )

    all_corrs = pl.concat(
        [
            sub.drop_nulls()
            .select(cs.float())
            .corr()
            .with_columns(
                pl.lit(case[0]).alias(dn.case_name),
                pl.Series(dn.label, sub.select(cs.float()).columns),
            )
            for case, sub in df.partition_by(dn.case_name, as_dict=True).items()
        ]
    )

    g = sns.catplot(
        data=all_corrs.filter(pl.col(dn.label) != qoi).to_pandas(),
        x=dn.label,
        y=qoi,
        kind="box",
        hue=dn.label,
    )
    return g.figure
