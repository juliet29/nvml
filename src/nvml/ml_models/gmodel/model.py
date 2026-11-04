import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GCNConv, HeteroConv, SAGEConv, global_mean_pool

from nvml.gmodel.hetero import FEATURE_DIM_OF, NODE_TYPES, RELATIONS
from nvml.gmodel.model_interfaces import GraphModelParams


class GCN(torch.nn.Module):
    def __init__(self, graph_params: GraphModelParams):
        super(GCN, self).__init__()
        torch.manual_seed(12345)
        hidden_channels = graph_params.hidden_channels
        self.conv1 = GCNConv(graph_params.num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, graph_params.num_classes)

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(
            x, p=0.5, training=self.training
        )  # multi-layer perceptron head? basically an ANN?
        x = self.lin(x)

        return x


class HeterogGCN(torch.nn.Module):
    def __init__(self, graph_params: GraphModelParams):
        super().__init__()
        torch.manual_seed(12345)
        hidden_channels = graph_params.hidden_channels

        def make_conv(in_dim_of: dict[str, int], out_channels: int):
            def conv_for(relation):
                src, _, dst = relation
                return SAGEConv((in_dim_of[src], in_dim_of[dst]), out_channels)

            return HeteroConv(
                {relation: conv_for(relation) for relation in RELATIONS}, aggr="sum"
            )

        self.conv1 = make_conv(FEATURE_DIM_OF, hidden_channels)
        self.conv2 = make_conv(
            {node_type: hidden_channels for node_type in NODE_TYPES}, hidden_channels
        )
        self.lin = Linear(hidden_channels, graph_params.num_classes)

    def forward(self, x_dict, edge_index_dict, batch_dict):
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {node_type: x.relu() for node_type, x in x_dict.items()}
        x_dict = self.conv2(x_dict, edge_index_dict)

        pooled = [
            global_mean_pool(x_dict[node_type], batch_dict[node_type])
            for node_type in NODE_TYPES
        ]
        x = torch.stack(pooled, dim=0).sum(dim=0)

        x = F.dropout(x, p=0.5, training=self.training)
        return self.lin(x)
