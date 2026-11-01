from cyclopts import App

from nvml._study.config.config import MakeConfig
from nvml._study.config.cstore import CONFIGS_DICT
from nvml.features.qoi import calc_qdim_quantity
from nvml.io import get_ambient_data_as_ds, read_graph
from nvml.qdim.wind import add_wind_sector_coord

qoi_app = App("qoi")

cfg = CONFIGS_DICT["msd_aug20"]
CASE_IX = 0


def get_ambient_ds(ix: int, cfg: MakeConfig):
    res = get_ambient_data_as_ds(cfg.get_one_case_data(ix).sql)
    res = add_wind_sector_coord(res)
    return res


@qoi_app.command()
def fe(case_ix: int = CASE_IX):
    ambient_ds = get_ambient_ds(case_ix, cfg)
    G = read_graph(cfg.get_one_graph_path(case_ix))
    da = calc_qdim_quantity(G, ambient_ds, "N")
    print("hello")
    print(da.max())
    return da
