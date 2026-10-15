from cyclopts import App
from plan2eplus.ezcase.ez import EZ

from nvml.cli.config.cstore import CONFIGS_DICT
from nvml.cli.studies.helpers import get_ambient_ds
from nvml.constants import DataNames as dn
from nvml.io import read_graph
from nvml.qdim.incident import make_zone_outward_normal_da, wind_angle_da_to_vectors
from nvml.qdim.intext import get_normals_for_windows_across_zones

coinc_app = App("coinc")

cfg = CONFIGS_DICT["msd_aug20"]
CASE_IX = 0
case_name = cfg.get_one_case(CASE_IX)
case_data = cfg.get_one_case_data(CASE_IX)
case_graph_path = cfg.get_one_graph_path(CASE_IX)


@coinc_app.command()
def fc():
    ambient_ds = get_ambient_ds(CASE_IX)
    G = read_graph(case_graph_path)
    case = EZ(case_data.idf)

    zons = get_normals_for_windows_across_zones(G, case)
    zone_da = make_zone_outward_normal_da(zons)
    return zone_da
    wind_da = wind_angle_da_to_vectors(ambient_ds[dn.wind_dir])
