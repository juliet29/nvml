import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GCNConv, global_mean_pool

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
