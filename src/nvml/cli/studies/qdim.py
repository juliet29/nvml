from cyclopts import App

from nvml.cli.config import CONFIGS_DICT
from nvml.cli.studies.paths import ProjectPaths
from nvml.constants import DataNames as dn
from nvml.qdim.corr_plot import make_comparison_plot
from nvml.qdim.data import add_wind_sector_coord, get_ambient_data_as_ds, graph_to_ds

qdim = App("qdim")


cfg = CONFIGS_DICT["jun24"]


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

    path = cfg.make_json_path(cfg.get_one_case())
    qoi_ds = graph_to_ds(path)

    savedir = ProjectPaths.figs.qdim_corr
    make_comparison_plot(ambient_ds, qoi_ds, dn.wind_speed, savedir)
