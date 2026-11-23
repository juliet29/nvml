import polars as pl
from plan2eplus.ezcase.ez import EZ
from plyze.flow_graph.interfaces import FlowGraph

from nvml import dn
from nvml.df_names import dfc
from nvml.fexp.coinc import get_coinc_for_all_normal_drns
from nvml.fexp.constant import (
    calc_graph_theory_metrics,
    calc_plan_area_metrics,
)
from nvml.fexp.dom_node_all import calc_dom_node_flow_metrics_for_all_drns


def gather_wind_dependent_data(case: EZ, G: FlowGraph, case_name: str):
    assert G
    co = get_coinc_for_all_normal_drns(G, case)
    fm = calc_dom_node_flow_metrics_for_all_drns(G)

    rows = [
        {
            dfc.wind_sector: direction,
            **co[direction]._asdict(),
            **fm[direction]._asdict(),
        }
        for direction in co
    ]
    df = pl.DataFrame(rows).with_columns(pl.lit(case_name).alias(dn.case_name))

    # df = (
    #     pl.DataFrame([{dn.wind_sector: k, **v._asdict()} for k, v in co.items()])
    #     .with_columns(pl.col(dn.wind_sector).replace(degrees).alias(dn.degree_dom_node))
    #     .with_columns(pl.lit(case_name).alias(dn.case_name))
    # )
    return df


def gather_constant_data(G: FlowGraph, case_name: str):
    bp = calc_plan_area_metrics(G)
    gt = calc_graph_theory_metrics(G)
    res = {dn.case_name: case_name} | bp._asdict() | gt._asdict()
    return res
