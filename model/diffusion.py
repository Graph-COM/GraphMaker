import numpy as np
import torch
import torch.nn as nn

from .gnn import *

__all__ = ["ModelSync"]

class MarginalTransition(nn.Module):
    """
    Parameters
    ----------
    device : torch.device
    X_marginal : torch.Tensor of shape (F, 2)
        X_marginal[f, :] is the marginal distribution of the f-th node attribute.
    E_marginal : torch.Tensor of shape (2)
        Marginal distribution of the edge existence.
    num_classes_E : int
        Number of edge classes.
    """
    def __init__(self,
                 device,
                 X_marginal,
                 E_marginal,
                 num_classes_E):
        super().__init__()

        num_attrs_X, num_classes_X = X_marginal.shape
        # (F, 2, 2)
        self.I_X = torch.eye(num_classes_X, device=device).unsqueeze(0).expand(
            num_attrs_X, num_classes_X, num_classes_X).clone()
        # (2, 2)
        self.I_E = torch.eye(num_classes_E, device=device)

        # (F, 2, 2)
        self.m_X = X_marginal.unsqueeze(1).expand(
            num_attrs_X, num_classes_X, -1).clone()
        # (2, 2)
        self.m_E = E_marginal.unsqueeze(0).expand(num_classes_E, -1).clone()

        self.I_X = nn.Parameter(self.I_X, requires_grad=False)
        self.I_E = nn.Parameter(self.I_E, requires_grad=False)

        self.m_X = nn.Parameter(self.m_X, requires_grad=False)
        self.m_E = nn.Parameter(self.m_E, requires_grad=False)

class NoiseSchedule(nn.Module):
    """
    Parameters
    ----------
    T : int
        Number of diffusion time steps.
    device : torch.device
    s : float
        Small constant for numerical stability.
    """
    def __init__(self, T, device, s=0.008):
        super().__init__()

        # Cosine schedule as proposed in
        # https://arxiv.org/abs/2102.09672
        num_steps = T + 2
        t = np.linspace(0, num_steps, num_steps)
        # Schedule for \bar{alpha}_t = alpha_1 * ... * alpha_t
        alpha_bars = np.cos(0.5 * np.pi * ((t / num_steps) + s) / (1 + s)) ** 2
        # Make the largest value 1.
        alpha_bars = alpha_bars / alpha_bars[0]
        alphas = alpha_bars[1:] / alpha_bars[:-1]

        self.betas = torch.from_numpy(1 - alphas).float().to(device)
        self.alphas = 1 - torch.clamp(self.betas, min=0, max=0.9999)

        log_alphas = torch.log(self.alphas)
        log_alpha_bars = torch.cumsum(log_alphas, dim=0)
        self.alpha_bars = torch.exp(log_alpha_bars)

        self.betas = nn.Parameter(self.betas, requires_grad=False)
        self.alphas = nn.Parameter(self.alphas, requires_grad=False)
        self.alpha_bars = nn.Parameter(self.alpha_bars, requires_grad=False)

class LossE(nn.Module):
    def __init__(self):
        super().__init__()

class BaseModel(nn.Module):
    """
    Parameters
    ----------
    T : int
        Number of diffusion time steps.
    X_marginal : torch.Tensor of shape (F, 2)
        X_marginal[f, :] is the marginal distribution of the f-th node attribute.
    Y_marginal : torch.Tensor of shape (C)
        Marginal distribution of the node labels.
    E_marginal : torch.Tensor of shape (2)
        Marginal distribution of the edge existence.
    num_nodes : int
        Number of nodes in the original graph.
    """
    def __init__(self,
                 T,
                 X_marginal,
                 Y_marginal,
                 E_marginal,
                 num_nodes):
        super().__init__()

        device = X_marginal.device
        # 2 for if edge exists or not.
        self.num_classes_E = 2
        self.num_attrs_X, self.num_classes_X = X_marginal.shape
        self.num_classes_Y = len(Y_marginal)

        self.transition = MarginalTransition(device, X_marginal,
                                             E_marginal, self.num_classes_E)

        self.T = T
        # Number of intermediate time steps to use for validation.
        self.num_denoise_match_samples = self.T - 1
        self.noise_schedule = NoiseSchedule(T, device)

        self.num_nodes = num_nodes

        self.X_marginal = X_marginal
        self.Y_marginal = Y_marginal
        self.E_marginal = E_marginal

        self.loss_E = LossE()

class LossX(nn.Module):
    """
    Parameters
    ----------
    num_attrs_X : int
        Number of node attributes.
    num_classes_X : int
        Number of classes for each node attribute.
    """
    def __init__(self,
                 num_attrs_X,
                 num_classes_X):
        super().__init__()

        self.num_attrs_X = num_attrs_X
        self.num_classes_X = num_classes_X

class ModelSync(BaseModel):
    """
    Parameters
    ----------
    T : int
        Number of diffusion time steps.
    X_marginal : torch.Tensor of shape (F, 2)
        X_marginal[f, :] is the marginal distribution of the f-th node attribute.
    Y_marginal : torch.Tensor of shape (C)
        Marginal distribution of the node labels.
    E_marginal : torch.Tensor of shape (2)
        Marginal distribution of the edge existence.
    gnn_X_config : dict
        Configuration of the GNN for reconstructing node attributes.
    gnn_E_config : dict
        Configuration of the GNN for reconstructing edges.
    num_nodes : int
        Number of nodes in the original graph.
    """
    def __init__(self,
                 T,
                 X_marginal,
                 Y_marginal,
                 E_marginal,
                 gnn_X_config,
                 gnn_E_config,
                 num_nodes):
        super().__init__(T=T,
                         X_marginal=X_marginal,
                         Y_marginal=Y_marginal,
                         E_marginal=E_marginal,
                         num_nodes=num_nodes)

        self.graph_encoder = GNN(num_attrs_X=self.num_attrs_X,
                                 num_classes_X=self.num_classes_X,
                                 num_classes_Y=self.num_classes_Y,
                                 num_classes_E=self.num_classes_E,
                                 gnn_X_config=gnn_X_config,
                                 gnn_E_config=gnn_E_config)

        self.loss_X = LossX(self.num_attrs_X, self.num_classes_X)

    def log_p_t(self,
                X_one_hot_3d,
                E_one_hot,
                Y,
                batch_src,
                batch_dst,
                batch_E_one_hot,
                t=None):
        """Obtain G^t from G^0 and compute log p(G^0 | G^t, Y, t).

        Parameters
        ----------
        X_one_hot_3d : torch.Tensor of shape (F, |V|, 2)
            X_one_hot_3d[f, :, :] is the one-hot encoding of the f-th node attribute.
        E_one_hot : torch.Tensor of shape (|V|, |V|, 2)
            - E_one_hot[:, :, 0] indicates the absence of an edge.
            - E_one_hot[:, :, 1] is the original adjacency matrix.
        Y : torch.Tensor of shape (|V|)
            Categorical node labels.
        batch_src : torch.LongTensor of shape (B)
            Source node IDs for a batch of edges (node pairs).
        batch_dst : torch.LongTensor of shape (B)
            Destination node IDs for a batch of edges (node pairs).
        batch_E_one_hot : torch.Tensor of shape (B, 2)
            batch_E_one_hot[batch_dst, batch_src].
        t : torch.LongTensor of shape (1), optional
            If specified, a time step will be enforced rather than sampled.

        Returns
        -------
        """
