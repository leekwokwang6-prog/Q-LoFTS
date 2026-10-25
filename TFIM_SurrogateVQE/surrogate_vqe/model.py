"""Fourier surrogate architecture preserved from the original training script."""

import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


def set_seed(seed):
    """Preserve the original Python, NumPy, PyTorch, and cuDNN seed setup."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed set to: {seed}")


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
        self.fourier_layer = FourierProductLayer(
            input_dim, hidden_units, max_freq,
            use_geometric_mean=(input_dim > 10)
        )
        self.dropout = nn.Dropout(p=dropout_rate)
        self.output_head = nn.Linear(hidden_units, 1)
        nn.init.normal_(self.output_head.weight, std=0.1)

    def forward(self, x):
        feat_fourier = self.fourier_layer(x)
        combined = feat_fourier
        combined = self.dropout(combined)
        out = self.output_head(combined)
        return out


def total_param_norm(model):
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5


def infer_architecture(state_dict, dropout_rate=0.0):
    """Read the model shape without constructing a model or consuming RNG."""
    shape = tuple(state_dict["fourier_layer.a_coeffs"].shape)
    if len(shape) != 3 or min(shape) < 1:
        raise ValueError("Invalid Fourier coefficient shape in checkpoint.")
    hidden_units, input_dim, max_freq = shape
    if tuple(state_dict["output_head.weight"].shape) != (1, hidden_units):
        raise ValueError("Inconsistent output head shape in checkpoint.")
    return {
        "input_dim": input_dim,
        "hidden_units": hidden_units,
        "max_freq": max_freq,
        "dropout_rate": float(dropout_rate),
    }


def load_model(checkpoint_path, dropout_rate=0.0):
    """Load a trusted state_dict, constructing the architecture exactly once."""
    state_dict = torch.load(Path(checkpoint_path), map_location="cpu")
    architecture = infer_architecture(state_dict, dropout_rate=dropout_rate)
    model = QLoFTSModel(**architecture)
    model.load_state_dict(state_dict)
    model.eval()
    return model, architecture
