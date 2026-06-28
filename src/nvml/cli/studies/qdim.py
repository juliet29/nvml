from cyclopts import App
from plan2eplus.ezcase.ez import EZ
from plyze import FlowGraphModel

from nvml.cli.config import CONFIGS_DICT
from nvml.cli.studies.paths import ProjectPaths
from nvml.constants import DataNames
from nvml.qdim.incident import (
    calculate_incidence_factor,
    make_zone_outward_normal_da,
    wind_angle_da_to_vectors,
)
from nvml.qdim.intext import get_normals_for_windows_across_zones
from nvml.qdim.io import get_ambient_data_as_ds, graph_to_ds
from nvml.qdim.wind import (
    add_incidence_data,
    add_wind_sector_coord,
    prep_comparison_data,
)

qdim = App("qdim")


cfg = CONFIGS_DICT["jun24"]


@qdim.command()
def fb():
    path = cfg.make_json_path(cfg.get_one_case())
    G = FlowGraphModel.read(path)
    return G
    # return make_int_ext_series(G)
    # sn = res.select("space_names").unique()
    # dt = res.select("datetimes").unique()
    # return res


@qdim.command()
def fc():
    path = cfg.make_json_path(cfg.get_one_case())
    res = graph_to_ds(path)
    # sn = res.select("space_names").unique()
    # dt = res.select("datetimes").unique()
    return res


@qdim.command()
def fd():
    res = get_ambient_data_as_ds(cfg.get_one_case_data().sql)
    res = add_wind_sector_coord(res)
    return res


@qdim.command()
def fe():
    res = get_ambient_data_as_ds(cfg.get_one_case_data().sql)
    ambient_ds = add_wind_sector_coord(res)

    json_path = cfg.make_json_path(cfg.get_one_case())
    qoi_ds = graph_to_ds(json_path)

    savedir = ProjectPaths.figs.qdim_corr

    G = FlowGraphModel.read(json_path)
    df = prep_comparison_data(G, ambient_ds, qoi_ds)
    return df


@qdim.command()
def ff():
    case_name = cfg.get_one_case(0)
    graph_path = cfg.make_json_path(case_name)
    idf_path = cfg.make_case_data(case_name).idf

    case = EZ(idf_path)
    G = FlowGraphModel.read(graph_path)
    zons = get_normals_for_windows_across_zones(G, case)
    zone_da = make_zone_outward_normal_da(zons)

    ds = get_ambient_data_as_ds(cfg.get_one_case_data().sql)
    wind_da = wind_angle_da_to_vectors(ds[DataNames.wind_dir])
    incidence_factor = calculate_incidence_factor(zone_da, wind_da)
    return incidence_factor


@qdim.command()
def ffa():
    ds = get_ambient_data_as_ds(cfg.get_one_case_data().sql)
    return wind_angle_da_to_vectors(ds[DataNames.wind_dir])


@qdim.command()
def ffb():
    case_name = cfg.get_one_case(0)
    graph_path = cfg.make_json_path(case_name)
    idf_path, sql_path = cfg.make_case_data(case_name)  # .idf
    case = EZ(idf_path)
    G = FlowGraphModel.read(graph_path)

    res = get_ambient_data_as_ds(sql_path)
    ambient_ds = add_wind_sector_coord(res)

    qoi_ds = graph_to_ds(graph_path)

    cd = prep_comparison_data(G, ambient_ds, qoi_ds)
    return add_incidence_data(G, case, ambient_ds, cd)
