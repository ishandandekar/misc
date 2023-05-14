import torch
from torch import nn

class LayerNormalization(nn.Module):
    def __init__(self, parameters_shape, eps=1e-5) -> None:
        super().__init__()
        self.parameters_shape = parameters_shape
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(self.parameters_shape))
        self.beta = nn.Parameter(torch.zeros(self.parameters_shape))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        dims = [-(i+1) for i in range(len(self.parameters_shape))]
        mean = inputs.mean(dim=dims, keepdim=True)
        var = ((inputs-mean) ** 2).mean(dim=dims, keepdim=True)
        std = (var + self.eps).sqrt()
        y = (inputs - mean) / std
        out = self.gamma * y + self.beta
        return out
