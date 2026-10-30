from pathlib import Path

import polars as pl
import seaborn.objects as so


class dfc:
    feature = "feature"
    target = "target"
    case_name = "case_name"


def plot_feature_v_target(df: pl.DataFrame, save_path: Path = Path("")):
    p = so.Plot(df, x=dfc.feature, y=dfc.target).add(so.Dot())
    return p


def plot_target_hist(df: pl.DataFrame):
    p = so.Plot(df, x=dfc.target).add(so.Bars(), so.Hist())
    return p
