from typing import NamedTuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer

# 1. Prepare synthetic data: y = 2x + 1
X = torch.randn(100, 1)  # 100 samples, 1 feature
Y = 2 * X + 1 + torch.randn(100, 1) * 0.1  # Target with some noise


# 2. Define the Linear Regression model
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 1 input feature, 1 output feature
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)


class ModelAndDetails(NamedTuple):
    model: nn.Module
    criterion: _Loss
    optimizer: Optimizer


class Data(NamedTuple):
    # TODO: should distinguish test and train datat with hard boundaries..
    X: torch.Tensor
    Y: torch.Tensor


def init_model():
    model = LinearRegressionModel()

    # 3. Define loss function and optimizer

    criterion = nn.MSELoss()  # Mean Squared Error
    optimizer = optim.SGD(model.parameters(), lr=0.01)  # Stochastic Gradient Descent

    return ModelAndDetails(model, criterion, optimizer)


def train_model(data: Data, mad: ModelAndDetails):
    # 4. Training loop
    epochs = 500
    for epoch in range(epochs):
        # Forward pass: compute predicted y
        pred_y = mad.model(data.X)

        # Compute loss
        loss = mad.criterion(pred_y, data.Y)

        # Zero gradients, perform backward pass, and update weights
        mad.optimizer.zero_grad()
        loss.backward()
        mad.optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")


def inspect_model(mad: ModelAndDetails):
    # 5. Inspect learned parameters
    [w, b] = mad.model.parameters()
    print(f"Learned Weight: {w[0][0].item():.4f}, Learned Bias: {b[0].item():.4f}")


def predict_with_model():
    pass
