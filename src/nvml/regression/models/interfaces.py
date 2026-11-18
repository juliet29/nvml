from plyze.qoi.xarray_helpers import CardinalNameLiteral

from nvml import dn
from nvml.qdim.wind import WindDirectionBinNames

direction_map: dict[WindDirectionBinNames, CardinalNameLiteral] = {
    "N": "NORTH",
    "E": "EAST",
    "S": "SOUTH",
    "W": "WEST",
}


class dfc:
    case_name = "case_name"

    wind_sector = dn.wind_sector

    # inflow =
    space_sum_time_mean = "space_sum_time_mean"

    area = "area"
    num_rooms = "num_rooms"
    num_paths = "num_paths"

    # paths
    avg_path_length = "avg_path_length"
    mode_path_length = "mode_path_length"
    len_longest_path = "len_longest_path"
    len_shortest_path = "len_shortest_path"
    degree_dom_node = "degree_dom_node"

    # --- added on later
    ww = "ww"
    lw = "lw"
    sd = "sd"
