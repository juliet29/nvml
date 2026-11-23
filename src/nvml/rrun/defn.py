from itertools import chain, combinations
from typing import Iterable

from nvml.nvml.df_names import dfc

# eventually this goes into public or a cli..


def powerset(iterable):
    "powerset([1,2,3]) -> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    if len(iterable) > 6:
        raise ValueError(f"{len(iterable)} > 6: going to produce more than 64 groups!")

    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(1, len(s) + 1))


def combine_features(features: Iterable[str]):
    return " + ".join(features)


def add_lhs(y: str, feature_string: str):
    return f"{y} ~ {feature_string}"


def simple():  # will return a list of ModelDefs
    # lets say want to test power set of a group on all qois
    # qois = dfc.get_qois()

    features = [dfc.area, dfc.degree_dom_node, dfc.avg_path_length]
    combos = powerset(features)
    strs = [combine_features(i) for i in combos]

    name = "intake_simple"

    return [add_lhs(dfc.intake, i) for i in strs], name
