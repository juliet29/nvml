import time
from pathlib import Path

import numpy as np
import polars as pl
from loguru import logger
from statsmodels.genmod.generalized_linear_model import GLMResultsWrapper
from tqdm import tqdm
from utils4plans.io import make_dir

from nvml.rrun.run import gamma_model


def time_id(step_seconds: int = 900) -> str:
    """Returns a 3-char monotonically increasing base-36 string (lasts ~16 months)."""
    EPOCH = 1795305600  # Nov 26
    t = int(time.time() - EPOCH) // step_seconds
    return np.base_repr(t, 36).lower().zfill(3)


class ModelFitData:
    defn = "defn"
    aic = "aic"
    bic = "bic"
    deviance = "deviance"
    null_deviance = "null_deviance"
    explained_deviance = "explained_deviance"
    pseudo_r2 = "pseudo_r2"


class CoeffData:
    feature = "feature"
    coff = "coef"
    std = "std_err"
    pval = "p_value"
    ci_lower = "ci_lower_95"
    ci_upper = "ci_upper_95"


def get_model_fit_data(defn: str, res: GLMResultsWrapper) -> dict:
    mfd = ModelFitData
    return {
        mfd.defn: defn,
        mfd.aic: res.aic,
        mfd.bic: res.bic_llf,
        mfd.deviance: res.deviance,
        mfd.null_deviance: res.null_deviance,
        mfd.explained_deviance: 1.0 - (res.deviance / res.null_deviance),
        mfd.pseudo_r2: res.pseudo_rsquared(),
    }


def get_coeff_data(res: GLMResultsWrapper):
    ci = res.conf_int()
    cd = CoeffData
    d = {
        cd.feature: res.params.index.to_list(),
        cd.coff: res.params.to_numpy(),
        cd.std: res.bse.to_numpy(),
        cd.pval: res.pvalues.to_numpy(),
        cd.ci_lower: ci[0].to_numpy(),
        cd.ci_upper: ci[1].to_numpy(),
    }
    return pl.DataFrame(d)


def fit_many_models(df: pl.DataFrame, defns: list[str]):
    models = [gamma_model(i, df) for i in defns]
    n = len(models)

    fitted_models = [
        m.fit() for m in tqdm(models, total=n, desc=f"Fitting models for {batch_name}")
    ]
    return fitted_models


def summary_save(df: pl.DataFrame, batch_name: str, defns: list[str], path: Path):
    # two tables, one with coeffs, one with evaluation data
    fitted_models = fit_many_models(df, defns)
    results = [
        get_model_fit_data(name, result) for name, result in zip(defns, fitted_models)
    ]
    result_df = pl.from_dicts(results)
    p = path / f"{batch_name}_{time_id()}.csv"
    make_dir(p)

    result_df.write_csv(p)
    logger.success(f"Saved to {p}")
    return p


def coeff_and_sumamry_save(
    df: pl.DataFrame, batch_name: str, defns: list[str], path: Path
):
    fitted_models = fit_many_models(df, defns)
    results = [
        get_model_fit_data(name, result) for name, result in zip(defns, fitted_models)
    ]
    result_df = pl.from_dicts(results)

    coeff_dfs = [
        get_model_coeffs(name, result) for name, result in zip(defns, fitted_models)
    ]
