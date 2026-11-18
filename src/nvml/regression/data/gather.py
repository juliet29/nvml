import polars as pl
from plan2eplus.ezcase.ez import EZ
from plyze.metrics.interfaces import FlowGraph

from nvml import dn
from nvml.regression.features.coinc import get_coinc_for_all_normal_drns
from nvml.regression.features.constant import (
    calc_flow_data_metrics,
    calc_plan_area_metrics,
)
from nvml.regression.features.dom_node import calc_degree_dominant_node_all_drns


def gather_wind_dependent_data(case: EZ, G: FlowGraph, case_name: str):
    assert G
    co = get_coinc_for_all_normal_drns(G, case)
    degrees = calc_degree_dominant_node_all_drns(G)

    df = (
        pl.DataFrame([{dn.wind_sector: k, **v._asdict()} for k, v in co.items()])
        .with_columns(pl.col(dn.wind_sector).replace(degrees).alias(dn.degree_dom_node))
        .with_columns(pl.lit(case_name).alias(dn.case_name))
    )
    return df


def gather_constant_data(G: FlowGraph, case_name: str):
    bp = calc_plan_area_metrics(G)
    fp = calc_flow_data_metrics(G)
    res = {dn.case_name: case_name} | bp._asdict() | fp._asdict()
    return res
