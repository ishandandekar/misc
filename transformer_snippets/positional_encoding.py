import torch
from torch import nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super().__init__()
        self.max_seq_length = max_seq_length
        self.d_model = d_model
        
    def forward(self):
        even_i = torch.arange(0, self.d_model, 2).float()
        denominator = torch.pow(10000, even_i/self.d_model)
        position = torch.arange(self.max_seq_length).reshape(self.max_seq_length, 1)
        even_PE = torch.sin(position / denominator)
        odd_PE = torch.cos(position / denominator)
        return torch.flatten(torch.stack([even_PE, odd_PE], dim=2), start_dim=1, end_dim=2)
