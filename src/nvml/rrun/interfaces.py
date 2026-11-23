import polars as pl
from statsmodels.genmod.generalized_linear_model import GLMResultsWrapper


class ModelFitData:
    defn = "defn"
    aic = "aic"
    bic = "bic"
    deviance = "deviance"
    null_deviance = "null_deviance"
    explained_deviance = "explained_deviance"
    pseudo_r2 = "pseudo_r2"


class CoeffData:
    defn = "defn"
    feature = "feature"
    coeff = "coef"
    std = "std_err"
    pval = "p_value"
    ci_lower = "ci_lower_95"
    ci_upper = "ci_upper_95"
    # rows added in by model
    intercept = "Intercept"


def get_model_summary_data(defn: str, res: GLMResultsWrapper) -> dict:
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


def get_coeff_data(defn: str, res: GLMResultsWrapper):
    ci = res.conf_int()
    cd = CoeffData
    d = {
        cd.feature: res.params.index.to_list(),
        cd.coeff: res.params.to_numpy(),
        cd.std: res.bse.to_numpy(),
        cd.pval: res.pvalues.to_numpy(),
        cd.ci_lower: ci[0].to_numpy(),
        cd.ci_upper: ci[1].to_numpy(),
    }
    return pl.DataFrame(d).insert_column(0, pl.lit(defn).alias(cd.defn))
