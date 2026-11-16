from nvml._paths.data_paths import DataPaths
from nvml.constants import DataNames as dn


def hello() -> str:
    return "Hello from nvml!"


__all__ = ["dn", "DataPaths"]
