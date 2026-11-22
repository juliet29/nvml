from typing import Optional

import pandas as pd
import polars as pl
import statsmodels.api as sm
import statsmodels.formula.api as smf
from loguru import logger
from statsmodels.genmod.generalized_linear_model import GLMResultsWrapper

from nvml.rrun.reg_data_prep import DataSpec
from nvml.rrun.reg_interfaces import (
    LINKS,
    ExperimentConfig,
    ExperimentResults,
    ModelMetrics,
    ModelSpec,
    extract_coefficients,
    extract_metrics,
)


def fit_gamma_model(
    spec: ModelSpec, df: pd.DataFrame, cluster_col: Optional[str] = None
) -> GLMResultsWrapper:
    family = sm.families.Gamma(LINKS[spec.link])
    model = smf.glm(spec.formula, data=df, family=family)
    if cluster_col:
        return model.fit(cov_type="cluster", cov_kwds={"groups": df[cluster_col]})
    return model.fit()


def run_experiment(
    config: ExperimentConfig, df: pl.DataFrame, data_spec: DataSpec
) -> ExperimentResults:
    logger.debug(df.columns)
    df_clean = df.to_pandas().dropna(subset=config.required_vars(df.columns)).copy()
    metrics_rows: list[ModelMetrics] = []
    coef_frames: list[pd.DataFrame] = []
    fitted: dict[str, GLMResultsWrapper] = {}

    for spec in config.model_specs:
        res = fit_gamma_model(spec, df_clean, config.cluster_col)
        fitted[spec.name] = res
        metrics_rows.append(extract_metrics(spec, res))
        coef_frames.append(extract_coefficients(spec, res))

    metrics_df = (
        pd.DataFrame(metrics_rows).sort_values("bic_llf").reset_index(drop=True)
    )
    coefs_df = pd.concat(coef_frames, ignore_index=True)
    return ExperimentResults(
        data_spec=data_spec,
        metrics=metrics_df,
        coefficients=coefs_df,
        fitted_models=fitted,
    )
