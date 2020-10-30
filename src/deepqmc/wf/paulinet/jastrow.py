from torch import nn

from deepqmc.torchext import SSP, get_log_dnn


class Jastrow(nn.Module):
    def __init__(self, embedding_dim, *, n_layers=3, sum_first=True, activation=SSP):
        super().__init__()
        self.dnn = get_log_dnn(
            embedding_dim, 1, activation, n_layers=n_layers, last_bias=True
        )
        self.operation = (
            lambda x: self.dnn(x.sum(dim=-2))
            if sum_first
            else lambda x: self.dnn(x).sum(dim=-2)
        )

    def forward(self, embedding):
        return self.operation(embedding).squeeze(dim=-1)
