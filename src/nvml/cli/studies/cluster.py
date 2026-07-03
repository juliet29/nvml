from cyclopts import App

from nvml.cli.studies.helpers import get_ambient_ds, get_graph_path
from nvml.cluster.assemble import make_space_name_by_wind_sector_da

cluster = App("cluster")


@cluster.command()
def fc():
    path = get_graph_path()
    ambient_ds = get_ambient_ds()
    return make_space_name_by_wind_sector_da(path, ambient_ds)
