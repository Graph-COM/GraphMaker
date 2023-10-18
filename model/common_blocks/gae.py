import torch.nn as nn

from .gcn import GCN

class GAE(nn.Module):
    def __init__(self,
                 in_size,
                 num_layers,
                 hidden_size,
                 dropout):
        super().__init__()

        self.gcn = GCN(in_size,
                       hidden_size,
                       num_layers,
                       hidden_size,
                       dropout)

    def forward(self, A, Z):
        Z = self.gcn(A, Z)
        return Z
