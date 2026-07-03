class DataNames:
    # properties
    is_external = "is_external"
    incident_factor = "incident_factor"

    # flow paths
    edge_name = "edge_name"
    u = "u"
    v = "v"

    # qois
    zone_dimless_flow = "zone_dimless_flow"
    zone_inflow = "zone_inflow"

    # ambient qois
    wind_dir = "wind_direction"
    wind_speed = "wind_speed"
    wind_sector = "wind_sector"
    t_out = "t_out"

    # coordinates
    case_name = "case_name"
    space_name = "space_names"
    space_ix = "space_ix"
    datetime = "datetimes"

    # incident angle calc
    xy_vector = "xy_vector"
    edge_num = "edge_num"


class FileNames:
    metrics_path = "cons/metrics.csv"
    qois_path = "cons/qois.csv"
    graph_json_path = "data/out.json"
