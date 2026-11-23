import time
from pathlib import Path

import numpy as np
import polars as pl
from loguru import logger
from tqdm import tqdm
from utils4plans.io import make_dir

from nvml.rrun.interfaces import get_coeff_data, get_model_summary_data
from nvml.rrun.run import gamma_model


def time_id(step_seconds: int = 900) -> str:
    """Returns a 3-char monotonically increasing base-36 string (lasts ~16 months)."""
    EPOCH = 1795305600  # Nov 26
    t = int(time.time() - EPOCH) // step_seconds
    return np.base_repr(t, 36).lower().zfill(3)


def fit_many_models(df: pl.DataFrame, defns: list[str]):
    models = [gamma_model(i, df) for i in defns]
    n = len(models)

    fitted_models = [m.fit() for m in tqdm(models, total=n, desc="Fitting models")]
    return fitted_models


def summary_save(df: pl.DataFrame, batch_name: str, defns: list[str], path: Path):
    fitted_models = fit_many_models(df, defns)
    summaries = [
        get_model_summary_data(name, result)
        for name, result in zip(defns, fitted_models)
    ]
    result_df = pl.from_dicts(summaries)
    p = path / f"{batch_name}_summary_{time_id()}.csv"
    make_dir(p)

    result_df.write_csv(p)
    logger.success(f"Saved to {p}")
    return p


def coeff_and_sumamry_save(
    df: pl.DataFrame, batch_name: str, defns: list[str], path: Path
):
    fitted_models = fit_many_models(df, defns)
    results = [
        get_model_summary_data(name, result)
        for name, result in zip(defns, fitted_models)
    ]
    result_df = pl.from_dicts(results)

    coeff_dfs = [
        get_coeff_data(name, result) for name, result in zip(defns, fitted_models)
    ]
    final_coeff_df = pl.concat(coeff_dfs, how="vertical")

    result_df = pl.from_dicts(results)
    p1 = path / f"{batch_name}_summary_{time_id()}.csv"
    p2 = path / f"{batch_name}_coeffs_{time_id()}.csv"
    make_dir(p1)

    result_df.write_csv(p1)
    final_coeff_df.write_csv(p2)
    logger.success(f"Saved to {p1} and {p2}")
    return p1, p2
