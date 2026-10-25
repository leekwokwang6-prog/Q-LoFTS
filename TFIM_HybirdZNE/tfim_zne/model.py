"""Fourier surrogate architecture used by the hybrid objective."""
import torch
import torch.nn as nn

class FourierProductLayer(nn.Module):

    def __init__(self, input_dim, hidden_units, max_freq, use_geometric_mean=True):
        super().__init__()
        self.d = input_dim
        self.H = hidden_units
        self.Lambda = max_freq
        self.use_geometric_mean = use_geometric_mean
        self.a_coeffs = nn.Parameter(torch.empty(hidden_units, input_dim, max_freq))
        self.b_coeffs = nn.Parameter(torch.empty(hidden_units, input_dim, max_freq))
        self.c_coeffs = nn.Parameter(torch.empty(hidden_units, input_dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.a_coeffs, mean=0.0, std=0.001)
        nn.init.normal_(self.b_coeffs, mean=0.0, std=0.001)
        nn.init.constant_(self.c_coeffs, 1.0)

    def forward(self, x):
        batch_size = x.shape[0]
        k_vec = torch.arange(1, self.Lambda + 1, device=x.device).float()
        x_expanded = x.unsqueeze(1).unsqueeze(-1)
        kx = x_expanded * k_vec
        term_cos = self.a_coeffs * torch.cos(kx)
        term_sin = self.b_coeffs * torch.sin(kx)
        fourier_sum = torch.sum(term_cos + term_sin, dim=-1)
        phi = self.c_coeffs + fourier_sum
        y_hidden = torch.prod(phi, dim=2)
        return y_hidden

class QLoFTSModel(nn.Module):

    def __init__(self, input_dim, hidden_units=32, max_freq=4, dropout_rate=0.2):
        super().__init__()
        self.fourier_layer = FourierProductLayer(input_dim, hidden_units, max_freq, use_geometric_mean=input_dim > 10)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.output_head = nn.Linear(hidden_units, 1)
        nn.init.normal_(self.output_head.weight, std=0.1)

    def forward(self, x):
        feat_fourier = self.fourier_layer(x)
        combined = feat_fourier
        combined = self.dropout(combined)
        out = self.output_head(combined)
        return out
