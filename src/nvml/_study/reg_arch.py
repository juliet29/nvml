import matplotlib
import polars as pl
from icecream import ic

from nvml.features.data_maker import dfc

matplotlib.use("module://matplotlib-backend-kitty")

from nvml._study._root.study_paths import StudyPaths
from nvml.constants import FileNames


def get_data():
    d = pl.read_csv(StudyPaths.data.anova.aug20v2 / FileNames.regression_data)
    return d


def get_data_and_splits():
    d = get_data()

    ww = [1 if "ww" in i else 0 for i in d[dfc.coincidence]]
    lw = [1 if "lw" in i else 0 for i in d[dfc.coincidence]]
    sd = [1 if "sd" in i else 0 for i in d[dfc.coincidence]]
    return d.with_columns(ww=pl.Series(ww), lw=pl.Series(lw), sd=pl.Series(sd))


# reporting ---
def fd():
    data = pl.read_csv(StudyPaths.data.anova.aug20 / FileNames.regression_data)
    result = data.select(pl.col(dfc.coincidence).value_counts(sort=True)).unnest(
        dfc.coincidence
    )
    ic(result)
