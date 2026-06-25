"""Graph autoencoder for embedding + clustering nvflow ventilation graphs.

Goal: learn one vector per graph whose geometry is *explainable* in terms of the
building's structure and size, then cluster those vectors.

Architecture (see memory/gae-design-decisions.md):
  - **Encoder**: GCN message passing -> node embeddings. Two variants:
      * deterministic  (wrapped by PyG ``GAE``)   -- simplest baseline / plumbing check.
      * variational    (wrapped by PyG ``VGAE``)  -- the real target: the KL term
        shapes a smooth, clusterable latent space and guards against the overfitting
        already seen on the tabular model.
  - **Edge decoder**: inner product (PyG default) -> reconstructs the adjacency
    (which rooms connect, which façade each room opens to).
  - **Feature decoders** (#4): small MLPs that reconstruct ONLY the structural/size
    features -- node area + aspect_ratio (zones only) and edge surface_area (windows).
    Without this, the latent has no incentive to remember *size*, one of our two
    clustering axes.
  - **Readout**: ``global_add_pool`` (SUM) -> graph vector. Sum (not mean) so that
    size / node-count survives into the embedding; mean-pool would wash it out.

Total loss:
    L = L_edge_recon  +  lambda_feat * (L_node_feat + L_edge_feat)  [ + beta * KL ]

References (further reading)
----------------------------
- Kipf & Welling, "Variational Graph Auto-Encoders", 2016. https://arxiv.org/abs/1611.07308
- Kipf & Welling, "Semi-Supervised Classification with GCNs", 2017. https://arxiv.org/abs/1609.02907
- Xu et al., "How Powerful are GNNs?" (GIN; sum-readout expressivity), 2019.
    https://arxiv.org/abs/1810.00826
- PyG autoencoder example this mirrors:
    https://github.com/pyg-team/pytorch_geometric/blob/master/examples/autoencoder.py
- PyG ``GAE`` / ``VGAE`` API:
    https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.models.GAE.html
- v2 idea -- physics-informed attention (flow as edge feature into attention):
    Veličković et al., GAT, https://arxiv.org/abs/1710.10903 ;
    Brody et al., GATv2, https://arxiv.org/abs/2105.14491 ;
    Shi et al., TransformerConv (edge features in attention), https://arxiv.org/abs/2009.03509
"""

import time
from dataclasses import dataclass
from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from loguru import logger
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GAE, VGAE, GCNConv, global_add_pool

from nvml.model.graph_data import (
    EDGE_RECON_IDX,
    F_IS_EXTERNAL,
    NODE_RECON_IDX,
)


class GCNEncoder(nn.Module):
    """Deterministic encoder: x -> node embeddings z (one vector per node)."""

    def __init__(self, in_dim: int, hidden: int, latent: int):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden)
        self.conv2 = GCNConv(hidden, latent)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)


class VariationalGCNEncoder(nn.Module):
    """Variational encoder: returns (mu, logstd); VGAE samples z = mu + eps*std."""

    def __init__(self, in_dim: int, hidden: int, latent: int):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden)
        self.conv_mu = GCNConv(hidden, latent)
        self.conv_logstd = GCNConv(hidden, latent)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)


class LossParts(NamedTuple):
    total: torch.Tensor
    edge: torch.Tensor
    node_feat: torch.Tensor
    edge_feat: torch.Tensor
    kl: torch.Tensor


class GraphAE(nn.Module):
    """Composes a PyG GAE/VGAE core with the two feature decoders + sum readout."""

    def __init__(
        self,
        in_dim: int,
        hidden: int = 64,
        latent: int = 16,
        variational: bool = True,
    ):
        super().__init__()
        self.variational = variational
        if variational:
            self.core = VGAE(VariationalGCNEncoder(in_dim, hidden, latent))
        else:
            self.core = GAE(GCNEncoder(in_dim, hidden, latent))

        # Feature decoders -- structural/size targets only (#4).
        self.node_decoder = nn.Sequential(
            nn.Linear(latent, hidden), nn.ReLU(), nn.Linear(hidden, len(NODE_RECON_IDX))
        )
        self.edge_decoder = nn.Sequential(
            nn.Linear(2 * latent, hidden), nn.ReLU(), nn.Linear(hidden, len(EDGE_RECON_IDX))
        )

    def loss(self, data, lambda_feat: float = 1.0, beta: float = 1.0) -> LossParts:
        # encode() samples for VGAE, is deterministic for GAE.
        z = self.core.encode(data.x, data.edge_index)

        # 1) adjacency reconstruction (inner-product decoder + negative sampling).
        edge = self.core.recon_loss(z, data.edge_index)

        # 2) KL regularization (VGAE only). Normalize by #nodes per the PyG example.
        kl = torch.zeros((), device=z.device)
        if self.variational:
            kl = self.core.kl_loss() / data.num_nodes

        # 3) node feature reconstruction -- ZONES ONLY (externals are zero-padded).
        zone = data.x[:, F_IS_EXTERNAL] == 0
        node_pred = self.node_decoder(z[zone])
        node_target = data.x[zone][:, NODE_RECON_IDX]
        node_feat = F.mse_loss(node_pred, node_target)

        # 4) edge feature reconstruction -- surface_area (window size) from endpoints.
        u, v = data.edge_index
        edge_pred = self.edge_decoder(torch.cat([z[u], z[v]], dim=-1))
        edge_target = data.edge_attr[:, EDGE_RECON_IDX]
        edge_feat = F.mse_loss(edge_pred, edge_target)

        total = edge + beta * kl + lambda_feat * (node_feat + edge_feat)
        return LossParts(total, edge.detach(), node_feat.detach(), edge_feat.detach(), kl.detach())

    @torch.no_grad()
    def embed(self, data) -> torch.Tensor:
        """One vector per graph for clustering. Uses mu (no sampling) for stable
        embeddings, then SUM-pools nodes within each graph."""
        if self.variational:
            z, _ = self.core.encoder(data.x, data.edge_index)
        else:
            z = self.core.encode(data.x, data.edge_index)
        batch = getattr(data, "batch", torch.zeros(data.num_nodes, dtype=torch.long))
        return global_add_pool(z, batch)


class ModelAndDetails(NamedTuple):
    model: GraphAE
    optimizer: optim.Optimizer


def init_graph_model(in_dim: int, variational: bool = True, lr: float = 1e-3) -> ModelAndDetails:
    model = GraphAE(in_dim=in_dim, variational=variational)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    return ModelAndDetails(model, optimizer)


@dataclass
class GraphTracker:
    total: float = 0.0
    edge: float = 0.0
    node_feat: float = 0.0
    edge_feat: float = 0.0
    kl: float = 0.0
    n: int = 0

    def reset(self):
        self.__init__()

    def update(self, lp: LossParts, n_graphs: int):
        self.total += lp.total.item() * n_graphs
        self.edge += lp.edge.item() * n_graphs
        self.node_feat += lp.node_feat.item() * n_graphs
        self.edge_feat += lp.edge_feat.item() * n_graphs
        self.kl += lp.kl.item() * n_graphs
        self.n += n_graphs

    def log(self, epoch, epochs, elapsed):
        if (epoch + 1) % 10 == 0:
            d = self.n or 1
            logger.info(
                f"Epoch [{epoch + 1}/{epochs}] loss: {self.total / d:.4f} "
                f"edge: {self.edge / d:.4f} node_feat: {self.node_feat / d:.4f} "
                f"edge_feat: {self.edge_feat / d:.4f} kl: {self.kl / d:.4f} "
                f"time: {elapsed * 1000:.1f}ms"
            )


def train_graph_model(
    train: list,
    mad: ModelAndDetails,
    epochs: int = 200,
    batch_size: int = 32,
    lambda_feat: float = 1.0,
    beta: float = 1.0,
) -> None:
    loader = DataLoader(train, batch_size=batch_size, shuffle=True)
    mad.model.train()
    mt = GraphTracker()

    for epoch in range(epochs):
        mt.reset()
        start = time.perf_counter()
        for batch in loader:
            lp = mad.model.loss(batch, lambda_feat=lambda_feat, beta=beta)
            mad.optimizer.zero_grad()
            lp.total.backward()
            mad.optimizer.step()
            mt.update(lp, batch.num_graphs)
        mt.log(epoch, epochs, time.perf_counter() - start)


@torch.no_grad()
def embed_graphs(graphs: list, mad: ModelAndDetails) -> torch.Tensor:
    """Stack per-graph embeddings -> (n_graphs, latent) matrix to cluster downstream
    (e.g. KMeans / agglomerative on these vectors; see scikit-learn clustering)."""
    mad.model.eval()
    loader = DataLoader(graphs, batch_size=64)
    return torch.cat([mad.model.embed(batch) for batch in loader], dim=0)
