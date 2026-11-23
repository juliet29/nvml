from pathlib import Path

import joblib
import polars as pl
import statsmodels.api as sm
import statsmodels.formula.api as smf
from loguru import logger
from utils4plans.io import make_dir


def gamma_model(model_def: str, df: pl.DataFrame):
    logger.info(f"Making model for {model_def}")
    gamma_family = sm.families.Gamma(link=sm.families.links.Log())
    model = smf.glm(formula=model_def, data=df, family=gamma_family)
    return model


def multi_model_save_joblib(
    batch_name: str, defs: list[str], models: list[sm.GLM], path: Path
):
    d = {d: m for d, m in zip(defs, models)}
    p = path / f"{batch_name}.joblib"
    make_dir(p)
    joblib.dump(d, p)
    logger.info(f"Saved to {p}")
