from pathlib import Path

import polars as pl

from nvml.constants import DataNames as dn


def collect_metrics_data(path: Path):
    df = pl.read_csv(path)
    return df

    # df_agg = df.select(c)
    # logger.debug(csv)


def collect_qoi_data(path: Path):
    dnq = dn.qois
    df = pl.read_csv(path)
    # take the median value across the sample times, the sum across the values for zones to produce one number per case
    df_agg = (
        df.group_by(dnq.case_name, dnq.space_name)
        .agg(pl.col(dnq.zone_dimless_flow).median())
        .group_by(dnq.case_name)
        .agg(pl.col(dnq.zone_dimless_flow).sum())
        # .select(dnq.zone_dimless_flow)
    )
    return df_agg


def arrange_data(metrics_path: Path, qois_path: Path):
    dnq = dn.qois
    metrics = collect_metrics_data(metrics_path)
    qoi = collect_qoi_data(qois_path)
    df = metrics.join(qoi, on=dnq.case_name)

    X = df.select(pl.exclude(dnq.case_name, dnq.zone_dimless_flow))
    y = df.select(dnq.zone_dimless_flow)
    return X, y


def df_to_torch(df: pl.DataFrame):
    pass
