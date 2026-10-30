import matplotlib
import polars as pl
import statsmodels.api as sm
import statsmodels.formula.api as smf
from icecream import ic

from nvml.features.visuals import plot_hist, plot_target_hist

matplotlib.use("module://matplotlib-backend-kitty")
from cyclopts import App
from nvml.features.data import save_data

from nvml.cli.config.cstore import CONFIGS_DICT
from nvml.cli.studies.study_paths import StudyPaths
from nvml.constants import FileNames
from nvml.features.visuals import dfc

reg_app = App("reg")


@reg_app.command()
def save_msd_data():
    cfg = CONFIGS_DICT["msd_aug20"]
    direction = "N"
    save_path = StudyPaths.data.anova.aug20
    return

    # "are you sure you want to overwerite this path?"
    save_data(cfg, direction, save_path)


@reg_app.command()
def fc():
    data = pl.read_csv(StudyPaths.data.anova.aug20 / FileNames.anova_data)
    p = plot_hist(data)
    p.show()


@reg_app.command()
def fcb():
    data = pl.read_csv(StudyPaths.data.anova.aug20 / FileNames.anova_data)
    p = plot_target_hist(data)
    p.show()


@reg_app.command()
def fd():
    data = pl.read_csv(StudyPaths.data.anova.aug20 / FileNames.anova_data)
    result = data.select(pl.col("feature").value_counts(sort=True)).unnest("feature")
    ic(result)


@reg_app.command()
def fe():
    data = pl.read_csv(StudyPaths.data.anova.aug20 / FileNames.anova_data)
    df = data.to_pandas()
    ic(df)
    # return
    mod = smf.ols(f"{dfc.target} ~ {dfc.feature}", df)
    ic(mod)
    res = mod.fit()
    table = sm.stats.anova_lm(res, typ=2)
    print(table)
