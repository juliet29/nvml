class MetricNames:
    pass


class QOINames:
    zone_dimless_flow = "zone_dimless_flow"
    wind_dir = "wind_direction"
    wind_sector = "wind_sector"


class VariableNames:
    space_name = "space_names"
    datetime = "datetimes"
    case_name = "case_name"


class DataNames:
    variables = VariableNames
    qois = QOINames


class FileNames:
    metrics_path = "cons/metrics.csv"
    qois_path = "cons/qois.csv"
    graph_json_path = "data/out.json"
