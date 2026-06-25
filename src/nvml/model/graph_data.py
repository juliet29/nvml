"""Build PyTorch Geometric graphs from nvflow ``out.json`` ventilation networks.

Each graph is one apartment's natural-ventilation airflow network:
  - *zone* nodes  : rooms (area, aspect_ratio, is_in_afn, ...)
  - *external* nodes : wind boundary conditions (NORTH/SOUTH/EAST/WEST), each
    carrying a compass direction and a wind-pressure series.
  - *edges* : external->zone (envelope openings; ``surface_area`` == window size)
    and zone->zone (internal doorways).

Design decisions encoded here (see memory/gae-design-decisions.md for the why):
  - v1 is **homogeneous with padded features** -- zones and externals share one
    node type, distinguished by an ``is_external`` flag. True ``HeteroData`` with
    separate node/edge types is the v2 target.
  - raw coordinates are intentionally **dropped** (position-invariant). Orientation
    enters *only* through external-node **bearing unit vectors** (sin, cos) -- so
    the model is position-invariant but orientation-aware. Two rooms with identical
    area + connectivity are deliberately indistinguishable (accepted caveat).
  - edges carry ``surface_area``; for external->zone edges this is the window size.
  - temporality is removed upstream by using dimensionless, wind-direction-normalized
    metrics (``zone_dimless_flow`` / ``zone_dimless_temp``): one static graph per
    (building, dominant wind direction). Loading those per-direction physics values
    onto the graph is stubbed below (depends on the still-open larger-model fusion
    decision -- building vs building x direction).

References (further reading)
----------------------------
- PyG ``Data`` / building your own graphs:
    https://pytorch-geometric.readthedocs.io/en/latest/get_started/introduction.html
- Heterogeneous graphs (the v2 target):
    https://pytorch-geometric.readthedocs.io/en/latest/tutorial/heterogeneous.html
- Why drop coordinates / invariances in GNNs (background on equivariance):
    Bronstein et al., "Geometric Deep Learning", 2021, https://arxiv.org/abs/2104.13478
"""

import json
from dataclasses import dataclass
from math import cos, radians, sin
from pathlib import Path

import numpy as np
import torch
from loguru import logger
from torch_geometric.data import Data

# --- Homogeneous (padded) node-feature layout -------------------------------
# One shared feature vector for every node; zones fill the geometry slots and
# externals fill the orientation slots, with an is_external flag to break ties.
#   zone     -> [area, aspect_ratio, is_in_afn,        0,        0, 0]
#   external -> [   0,            0,         0, bearing_sin, bearing_cos, 1]
F_AREA, F_ASPECT, F_AFN, F_BEARING_SIN, F_BEARING_COS, F_IS_EXTERNAL = range(6)
NODE_FEATURE_DIM = 6

# Feature-reconstruction targets (#4): ONLY the structural/size features, never
# the full vector. Nodes -> area + aspect_ratio (zones only); edges -> surface_area.
NODE_RECON_IDX = [F_AREA, F_ASPECT]
EDGE_RECON_IDX = [0]  # index into edge_attr (surface_area)

# --- Edge-feature layout ----------------------------------------------------
#   [surface_area, is_external_edge]  (+ optional dimensionless-flow channels, v2)
EDGE_FEATURE_DIM = 2

# Compass bearing in degrees, clockwise from North. Unit vector is (sin, cos) in
# (east, north) coordinates, so opposite directions are exact negatives:
#   N -> (0, 1)   E -> (1, 0)   S -> (0, -1)   W -> (-1, 0)
_COMPASS_DEG = {
    "NORTH": 0.0,
    "NORTHEAST": 45.0,
    "EAST": 90.0,
    "SOUTHEAST": 135.0,
    "SOUTH": 180.0,
    "SOUTHWEST": 225.0,
    "WEST": 270.0,
    "NORTHWEST": 315.0,
}


def bearing_vector(direction_name: str) -> tuple[float, float]:
    """(sin, cos) unit vector for a compass direction label like ``"EAST"``."""
    deg = _COMPASS_DEG[direction_name.strip().upper()]
    r = radians(deg)
    return sin(r), cos(r)


def _is_external(node_data: dict) -> bool:
    # Zone nodes have an "area"; external wind nodes do not (they carry pressure).
    return "area" not in node_data


def _node_features(name: str, data: dict) -> list[float]:
    if _is_external(data):
        bsin, bcos = bearing_vector(name)
        return [0.0, 0.0, 0.0, bsin, bcos, 1.0]
    return [
        float(data["area"]),
        float(data["aspect_ratio"]),
        float(data["is_in_afn"]),
        0.0,
        0.0,
        0.0,
    ]


def load_graph(graph_dir: Path, with_flow: bool = False) -> Data:
    """Parse one ``<graph_dir>/out.json`` into a homogeneous PyG ``Data`` object.

    Only the structural / geometric description is read here (no netCDF, no time
    series). Dimensionless per-direction flow is left to :func:`_load_dimless_flow`.
    """
    spec = json.loads((graph_dir / "out.json").read_text())

    names = [n["name"] for n in spec["nodes"]]
    index = {name: i for i, name in enumerate(names)}

    x = torch.tensor(
        [_node_features(n["name"], n["data"]) for n in spec["nodes"]],
        dtype=torch.float32,
    )

    # Undirected message passing: add each opening in both directions. The edge
    # set is also the positive-edge target the GAE/VGAE reconstructs.
    src, dst, attr = [], [], []
    for e in spec["edges"]:
        u, v = index[e["u"]], index[e["v"]]
        is_ext_edge = float(x[u, F_IS_EXTERNAL] == 1 or x[v, F_IS_EXTERNAL] == 1)
        feat = [float(e["data"]["surface_area"]), is_ext_edge]
        for a, b in ((u, v), (v, u)):
            src.append(a)
            dst.append(b)
            attr.append(feat)

    data = Data(
        x=x,
        edge_index=torch.tensor([src, dst], dtype=torch.long),
        edge_attr=torch.tensor(attr, dtype=torch.float32),
    )
    data.name = graph_dir.name

    if with_flow:
        _load_dimless_flow(graph_dir, data)
    return data


def _load_dimless_flow(graph_dir: Path, data: Data) -> None:
    """STUB: attach dimensionless, wind-direction-normalized flow to the graph.

    This is the physics signal for (a) condition-aware features and (b) the v2
    physics-based attention idea. Exactly how it lands on the graph depends on the
    still-open larger-model fusion decision:
      - building x direction  -> one graph per direction, single flow channel.
      - one building, multi-channel -> edge_attr gets [flow_N, flow_S, flow_E, flow_W]
        with masks for absent directions.
    Source is the per-zone/edge ``zone_dimless_flow`` (netCDF, needs xarray) or the
    aggregated ``cons/qois.csv`` already used by the classification pipeline.
    """
    raise NotImplementedError(
        "dimensionless-flow loading is pending the larger-model fusion decision"
    )


def load_dataset(graphs_root: Path, with_flow: bool = False) -> list[Data]:
    """Load every ``<graphs_root>/<id>/`` graph that has finished (``.done``)."""
    dirs = sorted(d for d in graphs_root.iterdir() if (d / "out.json").exists())
    graphs = [load_graph(d, with_flow=with_flow) for d in dirs]
    logger.info(f"Loaded {len(graphs)} graphs from {graphs_root}")
    return graphs


# --- Standardization --------------------------------------------------------
# Mirror the classification pipeline: fit scaling on TRAIN only, apply to both.
# Only the continuous structural features are standardized; flags (is_in_afn,
# is_external) and bearings (already in [-1, 1]) are left untouched.
_NODE_STD_IDX = [F_AREA, F_ASPECT]


@dataclass
class GraphStandardizer:
    node_mean: torch.Tensor
    node_std: torch.Tensor
    edge_mean: float
    edge_std: float

    @classmethod
    def fit(cls, train: list[Data]) -> "GraphStandardizer":
        # Zones only for node stats (externals are zero-padded in these columns).
        node_rows = torch.cat(
            [g.x[g.x[:, F_IS_EXTERNAL] == 0][:, _NODE_STD_IDX] for g in train]
        )
        edge_vals = torch.cat([g.edge_attr[:, 0] for g in train])
        s = cls(
            node_mean=node_rows.mean(0),
            node_std=node_rows.std(0).clamp_min(1e-8),
            edge_mean=float(edge_vals.mean()),
            edge_std=float(edge_vals.std().clamp_min(1e-8)),
        )
        logger.info(f"Fit standardizer: node_mean={s.node_mean.tolist()}")
        return s

    def apply(self, graphs: list[Data]) -> None:
        """Standardize in place. NOTE: x columns 0/1 then hold *scaled* values, so
        feature reconstruction (#4) targets the scaled space -- consistent."""
        cols = torch.tensor(_NODE_STD_IDX)
        for g in graphs:
            rows = (g.x[:, F_IS_EXTERNAL] == 0).nonzero(as_tuple=True)[0]
            g.x[rows[:, None], cols] = (g.x[rows[:, None], cols] - self.node_mean) / self.node_std
            g.edge_attr[:, 0] = (g.edge_attr[:, 0] - self.edge_mean) / self.edge_std


def train_test_split_graphs(
    graphs: list[Data], seed: int, train_size: float = 0.8
) -> tuple[list[Data], list[Data]]:
    """Random split at the *graph* level (the unit we cluster)."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(graphs))
    cut = int(train_size * len(graphs))
    train = [graphs[i] for i in perm[:cut]]
    test = [graphs[i] for i in perm[cut:]]
    logger.info(f"Split graphs -> train={len(train)} test={len(test)}")
    return train, test
