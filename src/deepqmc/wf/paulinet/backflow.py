import torch
from torch import nn

from deepqmc.torchext import SSP, get_log_dnn


class Backflow(nn.Module):
    def __init__(
        self, embedding_dim, n_orbitals, n_channels, *, n_layers=3, activation=SSP
    ):
        super().__init__()
        self.dnn = nn.ModuleList(
            [
                get_log_dnn(
                    embedding_dim,
                    n_orbitals,
                    activation,
                    n_layers=n_layers,
                    last_bias=True,
                )
                for _ in range(n_channels)
            ]
        )

    def forward(self, embedding):
        return torch.stack([bf(embedding) for bf in self.dnn], dim=1)
