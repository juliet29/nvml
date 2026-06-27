from cyclopts import App
from plyze import FlowGraphModel

from nvml.cli.config import CONFIGS_DICT
from nvml.cli.studies.paths import ProjectPaths
from nvml.qdim.intext import make_int_ext_series
from nvml.qdim.io import get_ambient_data_as_ds, graph_to_ds
from nvml.qdim.wind import add_wind_sector_coord, prep_comparison_data

qdim = App("qdim")


cfg = CONFIGS_DICT["jun24"]


@qdim.command()
def fb():
    path = cfg.make_json_path(cfg.get_one_case())
    G = FlowGraphModel.read(path)
    return make_int_ext_series(G)
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

    # plot(df, savedir, DataNames.t_out)
