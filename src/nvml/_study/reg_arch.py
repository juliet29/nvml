import matplotlib
import polars as pl
from icecream import ic

matplotlib.use("module://matplotlib-backend-kitty")

from nvml._study._root.study_paths import StudyPaths
from nvml.constants import FileNames


def get_data_and_splits():
    d = pl.read_csv(StudyPaths.data.anova.aug20 / FileNames.anova_data)

    ww = [1 if "ww" in i else 0 for i in d["feature"]]
    lw = [1 if "lw" in i else 0 for i in d["feature"]]
    sd = [1 if "sd" in i else 0 for i in d["feature"]]
    return d.with_columns(ww=pl.Series(ww), lw=pl.Series(lw), sd=pl.Series(sd))


def get_data():
    d = pl.read_csv(StudyPaths.data.anova.aug20 / FileNames.anova_data)
    return d


# reporting ---
def fd():
    data = pl.read_csv(StudyPaths.data.anova.aug20 / FileNames.anova_data)
    result = data.select(pl.col("feature").value_counts(sort=True)).unnest("feature")
    ic(result)
