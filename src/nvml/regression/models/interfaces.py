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

    # qoi
    intake = "intake"
    cv = "coeff_variation"
    dr = "dist"

    # bulk
    area = "area"
    num_rooms = "num_rooms"

    # paths
    num_paths = "num_paths"
    avg_path_length = "avg_path_length"
    mode_path_length = "mode_path_length"
    len_longest_path = "len_longest_path"
    len_shortest_path = "len_shortest_path"
    degree_dom_node = "degree_dom_node"

    # coincidence
    ww = "ww"
    lw = "lw"
    sd = "sd"

    @classmethod
    def pc(cls, v: int):
        return f"pc_{v}"

    @classmethod
    def all_params(cls, v: int):
        b = [cls.area, cls.num_rooms]
        p = [
            cls.num_paths,
            cls.avg_path_length,
            cls.mode_path_length,
            cls.len_longest_path,
            cls.len_shortest_path,
            cls.degree_dom_node,
        ]
        c = [cls.ww, cls.lw, cls.sd]
        pcs = [cls.pc(i) for i in range(v)]
        return b + p + c + pcs
