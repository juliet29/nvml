from typing import NamedTuple

from nvml.qdim.wind import WindDirectionBinNames


class ClusterModelParams(NamedTuple):
    wind_sector: WindDirectionBinNames
    n_clusters: int

    @property
    def tag(self):
        return f"{self.wind_sector}_{self.n_clusters}"
