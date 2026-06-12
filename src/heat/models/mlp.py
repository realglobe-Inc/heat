import torch.nn as nn
from torch.nn import functional


class MLP(nn.Module):
    """Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.output_dim = output_dim
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k)
            for n, k in zip([input_dim] + h, h + [output_dim], strict=False)
        )

    def forward(self, x):
        b, n, d = x.size()
        x = x.reshape(b * n, d)
        for i, layer in enumerate(self.layers):
            x = functional.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        x = x.view(b, n, self.output_dim)
        return x
