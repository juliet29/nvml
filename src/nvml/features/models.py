import polars as pl
import statsmodels.formula.api as smf

from nvml.features.data_maker import dfc

y = dfc.space_sum_time_mean


class Formulas:
    a = f"{y} ~ {dfc.num_paths}"
    b = f"{y} ~ {dfc.num_paths} + {dfc.degree_dom_node}"
    c = f"{y} ~ {dfc.num_paths} + {dfc.degree_dom_node} + {dfc.ww}"
    d = f"{y} ~ {dfc.num_paths} + {dfc.degree_dom_node} + {dfc.ww} + {dfc.area}"
    e = f"{y} ~ {dfc.num_paths} + {dfc.degree_dom_node} + {dfc.ww} + {dfc.area} + {dfc.num_rooms}"
    e1 = f"{y} ~ {dfc.num_paths} + {dfc.degree_dom_node} + {dfc.ww} + {dfc.lw} + {dfc.sd} + {dfc.area} + {dfc.num_rooms}"
    e2 = f"{y} ~ {dfc.num_paths} + {dfc.degree_dom_node}  + {dfc.lw} + {dfc.sd} + {dfc.area} + {dfc.num_rooms}"
    e3 = f"{y} ~ {dfc.num_paths} + {dfc.degree_dom_node}  + {dfc.lw} + {dfc.sd} + {dfc.area} + {dfc.num_rooms} + {dfc.len_longest_path} + {dfc.avg_path_length}"
    f = f"{y} ~ {dfc.num_paths} + {dfc.degree_dom_node} + {dfc.num_paths}:{dfc.degree_dom_node}"


def study_formula(data: pl.DataFrame, formula: str):
    df = data.filter(pl.col(dfc.ww) == 1).to_pandas()
    model = smf.ols(formula, data=df).fit()
    print("\033[1m" + formula + "\033[0m")
    print(model.summary())
    print("\n")
    print("\n")


def study_many_formulas(data: pl.DataFrame):
    ff = Formulas
    forms = [ff.a, ff.b, ff.c, ff.d, ff.e, ff.e1, ff.e2]
    forms = [ff.e2, ff.e3]
    # forms = [ff.f]
    for form in forms:
        study_formula(data, form)
