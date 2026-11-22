import datetime
import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, NamedTuple, Optional

import pandas as pd
import patsy
import statsmodels.api as sm
import yaml
from pydantic import BaseModel
from statsmodels.genmod.generalized_linear_model import GLMResultsWrapper

from nvml._paths.file_names import RegModelRunDir
from nvml.rrun.reg_data_prep import DataSpec

LINKS = {
    "log": sm.families.links.Log(),
    "inverse": sm.families.links.InversePower(),
    "identity": sm.families.links.Identity(),
}

LinkName = Literal["log", "inverse", "identity"]


def formula_variables(formula: str) -> set[str]:
    description = patsy.ModelDesc.from_formula(formula)
    terms = description.lhs_termlist + description.rhs_termlist
    return {factor.name() for term in terms for factor in term.factors}


class ModelSpec(BaseModel):
    name: str
    formula: str
    link: LinkName = "log"


class ModelGroup(BaseModel):
    name: str
    outcomes: str | list[str]
    formula_rhs: str
    link: LinkName = "log"

    def expand(self) -> list[ModelSpec]:
        outcomes = [self.outcomes] if isinstance(self.outcomes, str) else self.outcomes

        def spec_for(outcome: str) -> ModelSpec:
            name = f"{self.name}_{outcome}" if len(outcomes) > 1 else self.name
            return ModelSpec(
                name=name,
                formula=f"{outcome} ~ {self.formula_rhs}",
                link=self.link,
            )

        return [spec_for(outcome) for outcome in outcomes]


class ExperimentConfig(BaseModel):
    models: list[ModelGroup]
    cluster_col: Optional[str] = None

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ExperimentConfig":
        return cls.model_validate(yaml.safe_load(Path(path).read_text()))

    @property
    def model_specs(self) -> list[ModelSpec]:
        return [spec for group in self.models for spec in group.expand()]

    def required_vars(self, available: Iterable[str]) -> list[str]:
        referenced = set().union(
            *(formula_variables(spec.formula) for spec in self.model_specs)
        )
        available_set = set(available)
        return [name for name in referenced if name in available_set]


class ModelMetrics(NamedTuple):
    model: str
    formula: str
    link: str
    nobs: int
    bic_llf: float
    aic: float
    explained_deviance: float
    converged: bool


@dataclass
class ExperimentResults:
    data_spec: DataSpec

    metrics: pd.DataFrame
    coefficients: pd.DataFrame
    fitted_models: dict[str, GLMResultsWrapper] = field(repr=False)

    def save(self, base_output_dir: Path) -> Path:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = RegModelRunDir(
            base_output_dir / f"{timestamp}_{self.data_spec.run_slug()}"
        )
        run_dir.mkdir(parents=True, exist_ok=False)  # Fails loudly if collision occurs

        self.metrics.to_csv(run_dir.metrics, index=False)
        self.coefficients.to_csv(run_dir.coefficients, index=False)
        run_dir.data_spec.write_text(json.dumps(self.data_spec.to_dict(), indent=2))
        return run_dir

    # def save(self, output_dir: Path) -> None:
    #     output_dir.mkdir(parents=True, exist_ok=True)
    #     self.metrics.to_csv(output_dir / "model_comparison.csv", index=False)
    #     self.coefficients.to_csv(output_dir / "all_coefficients.csv", index=False)


def extract_metrics(spec: ModelSpec, res: GLMResultsWrapper) -> ModelMetrics:
    return ModelMetrics(
        model=spec.name,
        formula=spec.formula,
        link=spec.link,
        nobs=int(res.nobs),
        bic_llf=float(res.bic_llf),
        aic=float(res.aic),
        explained_deviance=float(1.0 - res.deviance / res.null_deviance),
        converged=bool(res.converged),
    )


def extract_coefficients(spec: ModelSpec, res: GLMResultsWrapper) -> pd.DataFrame:
    ci = res.conf_int()
    return pd.DataFrame(
        {
            "model": spec.name,
            "term": res.params.index,
            "coef": res.params.values,
            "std_err": res.bse.values,
            "p_value": res.pvalues.values,
            "ci_low": ci.iloc[:, 0].values,
            "ci_high": ci.iloc[:, 1].values,
        }
    )
