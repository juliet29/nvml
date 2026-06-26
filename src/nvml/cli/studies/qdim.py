from cyclopts import App

from nvml.cli.config import CONFIGS_DICT
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
