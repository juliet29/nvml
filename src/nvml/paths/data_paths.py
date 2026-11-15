from nvml.paths.file_names import Nov13KernDir, RegressionDir
from nvml.paths.settings import settings


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
    nov13 = Nov13KernDir(base / "nov13")


class DataPaths:
    gnn = GraphModels
    reg = RegressionModels
    kern = GraphKernels
