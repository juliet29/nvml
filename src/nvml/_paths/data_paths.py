from nvml._paths.file_names import GraphRepDir, RegressionDir
from nvml.settings import settings


class FlowGraphs:
    base = settings.path_to_data / "flow_graphs"
    test = base / "fg_test_1"
    jun24_3 = base / "fg_260624_3"
    jun30_50 = base / "fg_260630_50"
    aug20 = base / "fg_260820"


class GraphModels:
    base = settings.path_to_data / "gnn"
    test = base / "tests"
    case50 = base / "case50"
    aug20 = base / "aug20"


class RegressionModels:
    base = settings.path_to_data / "anova"
    aug20 = base / "aug20"
    aug20v2 = RegressionDir(base / "aug20v2")


class GraphKernels:
    base = settings.path_to_data / "kern"
    v1 = GraphRepDir(base / "v1")
    v2 = GraphRepDir(base / "v2")


class DataPaths:
    fg = FlowGraphs
    gnn = GraphModels
    reg = RegressionModels
    kern = GraphKernels
