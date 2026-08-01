from typing import NamedTuple

from nvml.qdim.wind import WindDirectionBinNames


class ClusterModelParams(NamedTuple):
    wind_sector: WindDirectionBinNames
    n_clusters: int
