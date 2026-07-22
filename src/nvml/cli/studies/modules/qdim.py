from cyclopts import App
from plan2eplus.ezcase.ez import EZ
from plyze import FlowGraphModel

from nvml.cli.config import CONFIGS_DICT, MakeConfig
from nvml.cli.studies.paths import ProjectPaths
from nvml.io import get_ambient_data_as_ds, graph_to_ds
from nvml.qdim.wind import (
    add_incidence_data,
    add_wind_sector_coord,
    prep_comparison_data,
)

qdim = App("qdim")

cfg = CONFIGS_DICT["jun24"]
CASE_IX = 0
CASE_NAME = cfg.get_one_case(CASE_IX)


@qdim.command()
def fe():
    res = get_ambient_data_as_ds(cfg.get_one_case_data().sql)
    ambient_ds = add_wind_sector_coord(res)

    json_path = cfg.make_graph_path(cfg.get_one_case())
    qoi_ds = graph_to_ds(json_path)

    savedir = ProjectPaths.figs.qdim_corr

    G = FlowGraphModel.read(json_path)
    df = prep_comparison_data(G, ambient_ds, qoi_ds)
    return df


def get_data_for_graph(cfg: MakeConfig, case_name: str):
    graph_path = cfg.make_graph_path(case_name)
    idf_path, sql_path = cfg.make_case_data(case_name)  # .idf
    case = EZ(idf_path)
    G = FlowGraphModel.read(graph_path)

    res = get_ambient_data_as_ds(sql_path)
    ambient_ds = add_wind_sector_coord(res)

    qoi_ds = graph_to_ds(graph_path)

    cd = prep_comparison_data(G, ambient_ds, qoi_ds)
    return add_incidence_data(G, case, ambient_ds, cd)


def load_config(name: str):
    return CONFIGS_DICT[name]
