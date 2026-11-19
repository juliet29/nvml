from itertools import product
from pathlib import Path

import numpy as np  # noqa
import pandas as pd
import polars as pl
import statsmodels.api as sm
import statsmodels.formula.api as smf
from loguru import logger

from nvml.qdim.wind import WindDirectionBinNames
from nvml.regression.models.collate import (
    RegDataPaths,
    collate_data,
    prepare_core_feature_df,
    prepare_pca_df,
)
from nvml.regression.models.interfaces import dfc
from nvml.wl.pca.pca_data import PCAId


def make_pca_dfs(path: Path, n: int):
    # pca_pairs = list(product(["wl", "sp"], ["int_ext", "area_ratio"]))
    pca_pairs = list(
        product(["wl", "sp"], ["int_ext", "area_ratio", "discrete_area_ratio"])
    )
    pca_dfs = {pair: prepare_pca_df(path, PCAId(*pair), n) for pair in pca_pairs}
    return pca_dfs


def make_core_feature_ds(rdp: RegDataPaths, drns: list[WindDirectionBinNames]):
    core_feature_dfs = {drn: prepare_core_feature_df(rdp, drn) for drn in drns}  # pyright: ignore[reportArgumentType]
    return core_feature_dfs


def prepare_model(
    df: pd.DataFrame,
    features: list[str],
    target: str,
):
    gamma_log = sm.families.Gamma(sm.families.links.Log())
    fstr = " + ".join(features)
    formula = f"{target} ~ {fstr}"
    model = smf.ols(formula, data=df, family=gamma_log).fit()

    return model


def prepare_many_models(
    qoi_df: pl.DataFrame,
    pca_dfs: dict,
    core_feature_dfs: dict,
    n: int,
    drns: list[WindDirectionBinNames],
):
    combos = list(product(drns, pca_dfs.keys()))

    def mk(drn, pca_tuple):
        k, la = pca_tuple
        name = f"{drn}_{k}_{la}"

        pca_df = pca_dfs[pca_tuple]
        base_df = core_feature_dfs[drn]
        df = collate_data(base_df, pca_df, qoi_df, drn).to_pandas()  # pyright: ignore[reportArgumentType]
        try:
            model = prepare_model(df, dfc.all_params(n), dfc.dr)

        except Exception as e:
            logger.critical(f"{e} for {name}")
            return name, {}
        logger.info(f"{name}: {model.rsquared}")
        return name, model

    res = [mk(drn, pca_tuple) for drn, pca_tuple in combos]
    rd = {i[0]: i[1] for i in res}
    return rd, core_feature_dfs
