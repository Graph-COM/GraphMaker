import torch.nn as nn
import torch.nn.functional as F

class GCN(nn.Module):
    def __init__(self,
                 in_size,
                 out_size,
                 num_layers,
                 hidden_size,
                 dropout):
        super().__init__()

        self.lins = nn.ModuleList()

        if num_layers >= 2:
            self.lins.append(nn.Linear(in_size, hidden_size))
            for _ in range(num_layers - 2):
                self.lins.append(nn.Linear(hidden_size, hidden_size))
            self.lins.append(nn.Linear(hidden_size, out_size))

        else:
            self.lins.append(nn.Linear(in_size, out_size))

        self.dropout = dropout

    def forward(self, A, H):
        for lin in self.lins[:-1]:
            H = A @ lin(H)
            H = F.relu(H)
            H = F.dropout(H, p=self.dropout, training=self.training)
        return A @ self.lins[-1](H)
