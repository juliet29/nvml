from typing import NamedTuple

from plan2eplus.geometry.directions import WallNormal

from nvml.qdim.angles import wind_angles_to_vector


class EdgeAndNormal(NamedTuple):
    u: str
    v: str
    normal_drn: WallNormal

    def __repr__(self) -> str:
        f = "EdgeAndNormal("
        f += f"({self.u, self.v}),\n"
        f += f"normal_drn={self.normal_drn},\n"
        f += f"vector={self.normal_as_vector}\n"
        f += ","
        return f

    @property
    def normal_as_vector(self):
        # multiply by -1 because want outward vector
        return -1 * wind_angles_to_vector(self.normal_drn.value)


class ZoneAndOutwardNormals(NamedTuple):
    zone_name: str
    edge_and_normals: list[EdgeAndNormal]

    @property
    def normal_vectors(self):
        return [i.normal_as_vector for i in self.edge_and_normals]
