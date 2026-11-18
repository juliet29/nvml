from typing import NamedTuple

from plan2eplus.ops.zones.ezobject import WallNormal

from nvml.qdim.wind import WindDirectionBinNames

direction_map: dict[WindDirectionBinNames, WallNormal] = {
    "N": WallNormal.NORTH,
    "E": WallNormal.EAST,
    "S": WallNormal.SOUTH,
    "W": WallNormal.WEST,
}


class TableEntry(NamedTuple):
    case_name: str
    space_mean_time_mean: float
    space_sum_time_mean: float
    coincidence: str
    area: float
    num_rooms: int
    num_paths: int
    avg_path_length: float
    mode_path_length: int
    len_longest_path: int
    len_shortest_path: int
    degree_dom_node: int


class dfc:
    case_name = "case_name"
    space_mean_time_mean = "space_mean_time_mean"
    space_sum_time_mean = "space_sum_time_mean"

    coincidence = "coincidence"

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
