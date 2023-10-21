import dgl.sparse as dglsp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from tqdm import tqdm

from .gnn import *

__all__ = ["ModelSync", "ModelAsync"]

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

    def get_Q_bar_E(self, alpha_bar_t):
        """Compute the probability transition matrices for obtaining A^t.

        Parameters
        ----------
        alpha_bar_t : torch.Tensor of shape (1)
            A value in [0, 1].

        Returns
        -------
        Q_bar_t_E : torch.Tensor of shape (2, 2)
            Transition matrix for corrupting graph structure at time step t.
        """
        Q_bar_t_E = alpha_bar_t * self.I_E + (1 - alpha_bar_t) * self.m_E

        return Q_bar_t_E

    def get_Q_bar_X(self, alpha_bar_t):
        """Compute the probability transition matrices for obtaining X^t.

        Parameters
        ----------
        alpha_bar_t : torch.Tensor of shape (1)
            A value in [0, 1].

        Returns
        -------
        Q_bar_t_X : torch.Tensor of shape (F, 2, 2)
            Transition matrix for corrupting node attributes at time step t.
        """
        Q_bar_t_X = alpha_bar_t * self.I_X + (1 - alpha_bar_t) * self.m_X

        return Q_bar_t_X

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

    def forward(self, true_E, logit_E):
        """
        Parameters
        ----------
        true_E : torch.Tensor of shape (B, 2)
            One-hot encoding of the edge existence for a batch of node pairs.
        logit_E : torch.Tensor of shape (B, 2)
            Predicted logits for the edge existence.

        Returns
        -------
        loss_E : torch.Tensor
            Scalar representing the loss for edge existence.
        """
        true_E = torch.argmax(true_E, dim=-1)    # (B)
        loss_E = F.cross_entropy(logit_E, true_E)

        return loss_E

class BaseModel(nn.Module):
    """
    Parameters
    ----------
    T : int
        Number of diffusion time steps - 1.
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
        self.num_denoise_match_samples = self.T
        self.noise_schedule = NoiseSchedule(T, device)

        self.num_nodes = num_nodes

        self.X_marginal = X_marginal
        self.Y_marginal = Y_marginal
        self.E_marginal = E_marginal

        self.loss_E = LossE()

    def sample_E(self, prob_E):
        """Sample a graph structure from prob_E.

        Parameters
        ----------
        prob_E : torch.Tensor of shape (|V|, |V|, 2)
            Probability distribution for edge existence.

        Returns
        -------
        E_t : torch.LongTensor of shape (|V|, |V|)
            Sampled symmetric adjacency matrix.
        """
        # (|V|^2, 1)
        E_t = prob_E.reshape(-1, prob_E.size(-1)).multinomial(1)

        # (|V|, |V|)
        num_nodes = prob_E.size(0)
        E_t = E_t.reshape(num_nodes, num_nodes)
        # Make it symmetric for undirected graphs.
        src, dst = torch.triu_indices(
            num_nodes, num_nodes, device=E_t.device)
        E_t[dst, src] = E_t[src, dst]

        return E_t

    def sample_X(self, prob_X):
        """Sample node attributes from prob_X.

        Parameters
        ----------
        prob_X : torch.Tensor of shape (F, |V|, 2)
            Probability distributions for node attributes.

        Returns
        -------
        X_t_one_hot : torch.Tensor of shape (|V|, 2 * F)
            One-hot encoding of the sampled node attributes.
        """
        # (F * |V|)
        X_t = prob_X.reshape(-1, prob_X.size(-1)).multinomial(1)
        # (F, |V|)
        X_t = X_t.reshape(self.num_attrs_X, -1)
        # (|V|, 2 * F)
        X_t_one_hot = torch.cat([
            F.one_hot(X_t[i], num_classes=self.num_classes_X)
            for i in range(self.num_attrs_X)
        ], dim=1).float()

        return X_t_one_hot

    def get_adj(self, E_t):
        """
        Parameters
        ----------
        E_t : torch.LongTensor of shape (|V|, |V|)
            Sampled symmetric adjacency matrix.

        Returns
        -------
        dglsp.SparseMatrix
            Row-normalized adjacency matrix.
        """
        # Row normalization.
        edges_t = E_t.nonzero().T
        num_nodes = E_t.size(0)
        A_t = dglsp.spmatrix(edges_t, shape=(num_nodes, num_nodes))
        D_t = dglsp.diag(A_t.sum(1)) ** -1
        return D_t @ A_t

    def denoise_match_Z(self,
                        Z_t_one_hot,
                        Q_t_Z,
                        Z_one_hot,
                        Q_bar_s_Z,
                        pred_Z):
        """Compute the denoising match term for Z given a
        sampled t, i.e., the KL divergence between q(D^{t-1}| D, D^t) and
        q(D^{t-1}| hat{p}^{D}, D^t).

        Parameters
        ----------
        Z_t_one_hot : torch.Tensor of shape (B, C) or (A, B, C)
            One-hot encoding of the data sampled at time step t.
        Q_t_Z : torch.Tensor of shape (C, C) or (A, C, C)
            Transition matrix from time step t - 1 to t.
        Z_one_hot : torch.Tensor of shape (B, C) or (A, B, C)
            One-hot encoding of the original data.
        Q_bar_s_Z : torch.Tensor of shape (C, C) or (A, C, C)
            Transition matrix from timestep 0 to t-1.
        pred_Z : torch.Tensor of shape (B, C) or (A, B, C)
            Predicted probs for the original data.

        Returns
        -------
        float
            KL value.
        """
        # q(Z^{t-1}| Z, Z^t)
        left_term = Z_t_one_hot @ torch.transpose(Q_t_Z, -1, -2) # (B, C) or (A, B, C)
        right_term = Z_one_hot @ Q_bar_s_Z                       # (B, C) or (A, B, C)
        product = left_term * right_term                         # (B, C) or (A, B, C)
        denom = product.sum(dim=-1)                              # (B,) or (A, B)
        denom[denom == 0.] = 1
        prob_true = product / denom.unsqueeze(-1)                # (B, C) or (A, B, C)

        # q(Z^{t-1}| hat{p}^{Z}, Z^t)
        right_term = pred_Z @ Q_bar_s_Z                          # (B, C) or (A, B, C)
        product = left_term * right_term                         # (B, C) or (A, B, C)
        denom = product.sum(dim=-1)                              # (B,) or (A, B)
        denom[denom == 0.] = 1
        prob_pred = product / denom.unsqueeze(-1)                # (B, C) or (A, B, C)

        # KL(q(Z^{t-1}| hat{p}^{Z}, Z^t) || q(Z^{t-1}| Z, Z^t))
        kl = F.kl_div(input=prob_pred.log(), target=prob_true, reduction='none')
        return kl.clamp(min=0).mean().item()

    def denoise_match_E(self,
                        t_float,
                        logit_E,
                        E_t_one_hot,
                        E_one_hot):
        """Compute the denoising match term for edge prediction given a
        sampled t, i.e., the KL divergence between q(D^{t-1}| D, D^t) and
        q(D^{t-1}| hat{p}^{D}, D^t).

        Parameters
        ----------
        t_float : torch.Tensor of shape (1)
            Sampled timestep divided by self.T.
        logit_E : torch.Tensor of shape (B, 2)
            Predicted logits for the edge existence of a batch of node pairs.
        E_t_one_hot : torch.Tensor of shape (B, 2)
            One-hot encoding of sampled edge existence for the batch of
            node pairs.
        E_one_hot : torch.Tensor of shape (B, 2)
            One-hot encoding of the original edge existence for the batch of
            node pairs.

        Returns
        -------
        float
            KL value.
        """
        t = int(t_float.item() * self.T)
        s = t - 1

        alpha_bar_s = self.noise_schedule.alpha_bars[s]
        alpha_t = self.noise_schedule.alphas[t]

        Q_bar_s_E = self.transition.get_Q_bar_E(alpha_bar_s)
        # Note that computing Q_bar_t from alpha_bar_t is the same
        # as computing Q_t from alpha_t.
        Q_t_E = self.transition.get_Q_bar_E(alpha_t)

        pred_E = logit_E.softmax(dim=-1)

        return self.denoise_match_Z(E_t_one_hot,
                                    Q_t_E,
                                    E_one_hot,
                                    Q_bar_s_E,
                                    pred_E)

    def posterior(self,
                  Z_t,
                  Q_t,
                  Q_bar_s,
                  Q_bar_t,
                  prior):
        """Compute the posterior distribution for time step s, i.e., t - 1.

        Parameters
        ----------
        Z_t : torch.Tensor of shape (B, 2) or (F, |V|, 2)
            One-hot encoding of the sampled data at timestep t.
            B for batch size, C for number of classes, D for number
            of features.
        Q_t : torch.Tensor of shape (2, 2) or (F, 2, 2)
            The transition matrix from timestep t-1 to t.
        Q_bar_s : torch.Tensor of shape (2, 2) or (F, 2, 2)
            The transition matrix from timestep 0 to t-1.
        Q_bar_t : torch.Tensor of shape (2, 2) or (F, 2, 2)
            The transition matrix from timestep 0 to t.
        prior : torch.Tensor of shape (B, 2) or (F, |V|, 2)
            Reconstructed prior distribution.

        Returns
        -------
        prob : torch.Tensor of shape (B, 2) or (D, B, C)
            Posterior distribution.
        """
        # (B, 2) or (F, |V|, 2)
        left_term = Z_t @ torch.transpose(Q_t, -1, -2)
        # (B, 1, 2) or (F, |V|, 1, 2)
        left_term = left_term.unsqueeze(dim=-2)
        # (1, 2, 2) or (F, 1, 2, 2)
        right_term = Q_bar_s.unsqueeze(dim=-3)
        # (B, 2, 2) or (F, |V|, 2, 2)
        # Different from denoise_match_z, this function does not
        # compute (Z_t @ Q_t.T) * (Z_0 @ Q_bar_s) for a specific
        # Z_0, but compute for all possible values of Z_0.
        numerator = left_term * right_term

        # (2, B) or (F, 2, |V|)
        prod = Q_bar_t @ torch.transpose(Z_t, -1, -2)
        # (B, 2) or (F, |V|, 2)
        prod = torch.transpose(prod, -1, -2)
        # (B, 2, 1) or (F, |V|, 2, 1)
        denominator = prod.unsqueeze(-1)
        denominator[denominator == 0.] = 1.
        # (B, 2, 2) or (F, |V|, 2, 2)
        out = numerator / denominator

        # (B, 2, 2) or (F, |V|, 2, 2)
        prob = prior.unsqueeze(-1) * out
        # (B, 2) or (F, |V|, C)
        prob = prob.sum(dim=-2)

        return prob

    def sample_E_infer(self, prob_E):
        """Draw a sample from prob_E

        Parameters
        ----------
        prob_E : torch.Tensor of shape (B, 2)
            Probability distributions for edge existence.

        Returns
        -------
        E_t : torch.LongTensor of shape (|V|, |V|)
            Sampled adjacency matrix.
        """
        E_t_ = prob_E.multinomial(1).squeeze(-1)
        E_t = torch.zeros(self.num_nodes, self.num_nodes).long().to(E_t_.device)
        E_t[self.dst, self.src] = E_t_
        E_t[self.src, self.dst] = E_t_

        return E_t

    def get_E_t(self,
                device,
                edge_data_loader,
                pred_E_func,
                t_float,
                X_t_one_hot,
                Y_0,
                E_t,
                Q_t_E,
                Q_bar_s_E,
                Q_bar_t_E,
                batch_size):
        A_t = self.get_adj(E_t)
        E_prob = torch.zeros(len(self.src), self.num_classes_E).to(device)

        start = 0
        for batch_edge_index in edge_data_loader:
            # (B, 2)
            batch_edge_index = batch_edge_index.to(device)
            batch_dst, batch_src = batch_edge_index.T
            # Reconstruct the edges.
            # (B, 2)
            batch_pred_E = pred_E_func(t_float,
                                       X_t_one_hot,
                                       Y_0,
                                       A_t,
                                       batch_src,
                                       batch_dst)

            batch_pred_E = batch_pred_E.softmax(dim=-1)

            # (B, 2)
            batch_E_t_one_hot = F.one_hot(
                E_t[batch_src, batch_dst],
                num_classes=self.num_classes_E).float()
            batch_E_prob_ = self.posterior(batch_E_t_one_hot, Q_t_E,
                                           Q_bar_s_E, Q_bar_t_E, batch_pred_E)

            end = start + batch_size
            E_prob[start: end] = batch_E_prob_
            start = end

        E_t = self.sample_E_infer(E_prob)

        return A_t, E_t

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

    def forward(self, true_X, logit_X):
        """
        Parameters
        ----------
        true_X : torch.Tensor of shape (F, |V|, 2)
            X_one_hot_3d[f, :, :] is the one-hot encoding of the f-th node attribute.
        logit_X : torch.Tensor of shape (|V|, F, 2)
            Predicted logits for the node attributes.

        Returns
        -------
        loss_X : torch.Tensor
            Scalar representing the loss for node attributes.
        """
        true_X = true_X.transpose(0, 1)               # (|V|, F, 2)
        # v1x1, v1x2, ..., v1xd, v2x1, ...
        true_X = true_X.reshape(-1, true_X.size(-1))  # (|V| * F, 2)

        # v1x1, v1x2, ..., v1xd, v2x1, ...
        logit_X = logit_X.reshape(true_X.size(0), -1) # (|V| * F, 2)

        true_X = torch.argmax(true_X, dim=-1)         # (|V| * F)
        loss_X = F.cross_entropy(logit_X, true_X)

        return loss_X

class ModelSync(BaseModel):
    """
    Parameters
    ----------
    T : int
        Number of diffusion time steps - 1.
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

    def apply_noise(self, X_one_hot_3d, E_one_hot, t=None):
        """Corrupt G and sample G^t.

        Parameters
        ----------
        X_one_hot_3d : torch.Tensor of shape (F, |V|, 2)
            X_one_hot_3d[f, :, :] is the one-hot encoding of the f-th node attribute
            in the real graph.
        E_one_hot : torch.Tensor of shape (|V|, |V|, 2)
            - E_one_hot[:, :, 0] indicates the absence of an edge in the real graph.
            - E_one_hot[:, :, 1] is the adjacency matrix of the real graph.
        t : torch.LongTensor of shape (1), optional
            If specified, a time step will be enforced rather than sampled.

        Returns
        -------
        t_float : torch.Tensor of shape (1)
            Sampled timestep divided by self.T.
        X_t_one_hot : torch.Tensor of shape (|V|, 2 * F)
            One-hot encoding of the sampled node attributes.
        E_t : torch.LongTensor of shape (|V|, |V|)
            Sampled symmetric adjacency matrix.
        """
        if t is None:
            # Sample a timestep t uniformly.
            # Note that the notation is slightly inconsistent with the paper.
            # t=0 corresponds to t=1 in the paper, where corruption has already taken place.
            t = torch.randint(low=0, high=self.T + 1, size=(1,),
                              device=X_one_hot_3d.device)

        alpha_bar_t = self.noise_schedule.alpha_bars[t]

        # Sample A^t.
        Q_bar_t_E = self.transition.get_Q_bar_E(alpha_bar_t) # (2, 2)
        prob_E = E_one_hot @ Q_bar_t_E                       # (|V|, |V|, 2)
        E_t = self.sample_E(prob_E)                          # (|V|, |V|)

        # Sample X^t.
        Q_bar_t_X = self.transition.get_Q_bar_X(alpha_bar_t) # (F, 2, 2)
        # Compute matrix multiplication over the first batch dimension.
        prob_X = torch.bmm(X_one_hot_3d, Q_bar_t_X)          # (F, |V|, 2)
        X_t_one_hot = self.sample_X(prob_X)

        t_float = t / self.T

        return t_float, X_t_one_hot, E_t

    def log_p_t(self,
                X_one_hot_3d,
                E_one_hot,
                Y,
                batch_src,
                batch_dst,
                batch_E_one_hot,
                t=None):
        """Obtain G^t and compute log p(G | G^t, Y, t).

        Parameters
        ----------
        X_one_hot_3d : torch.Tensor of shape (F, |V|, 2)
            X_one_hot_3d[f, :, :] is the one-hot encoding of the f-th node attribute
            in the real graph.
        E_one_hot : torch.Tensor of shape (|V|, |V|, 2)
            - E_one_hot[:, :, 0] indicates the absence of an edge in the real graph.
            - E_one_hot[:, :, 1] is the adjacency matrix of the real graph.
        Y : torch.Tensor of shape (|V|)
            Categorical node labels of the real graph.
        batch_src : torch.LongTensor of shape (B)
            Source node IDs for a batch of candidate edges (node pairs).
        batch_dst : torch.LongTensor of shape (B)
            Destination node IDs for a batch of candidate edges (node pairs).
        batch_E_one_hot : torch.Tensor of shape (B, 2)
            E_one_hot[batch_dst, batch_src].
        t : torch.LongTensor of shape (1), optional
            If specified, a time step will be enforced rather than sampled.

        Returns
        -------
        loss_X : torch.Tensor
            Scalar representing the loss for node attributes.
        loss_E : torch.Tensor
            Scalar representing the loss for edge existence.
        """
        t_float, X_t_one_hot, E_t = self.apply_noise(X_one_hot_3d, E_one_hot, t)
        A_t = self.get_adj(E_t)
        logit_X, logit_E = self.graph_encoder(t_float,
                                              X_t_one_hot,
                                              Y,
                                              A_t,
                                              batch_src,
                                              batch_dst)

        loss_X = self.loss_X(X_one_hot_3d, logit_X)
        loss_E = self.loss_E(batch_E_one_hot, logit_E)

        return loss_X, loss_E

    def denoise_match_X(self,
                        t_float,
                        logit_X,
                        X_t_one_hot,
                        X_one_hot_3d):
        """Compute the denoising match term for node attribute prediction given a
        sampled t, i.e., the KL divergence between q(D^{t-1}| D, D^t) and
        q(D^{t-1}| hat{p}^{D}, D^t).

        Parameters
        ----------
        t_float : torch.Tensor of shape (1)
            Sampled timestep divided by self.T.
        logit_X : torch.Tensor of shape (|V|, F, 2)
            Predicted logits for the node attributes.
        X_t_one_hot : torch.Tensor of shape (|V|, 2 * F)
            One-hot encoding of the node attributes sampled at time step t.
        X_one_hot_3d : torch.Tensor of shape (F, |V|, 2)
            X_one_hot_3d[f, :, :] is the one-hot encoding of the f-th node attribute.

        Returns
        -------
        float
            KL value for node attributes.
        """
        t = int(t_float.item() * self.T)
        s = t - 1

        alpha_bar_s = self.noise_schedule.alpha_bars[s]
        alpha_t = self.noise_schedule.alphas[t]

        Q_bar_s_X = self.transition.get_Q_bar_X(alpha_bar_s)
        # Note that computing Q_bar_t from alpha_bar_t is the same
        # as computing Q_t from alpha_t.
        Q_t_X = self.transition.get_Q_bar_X(alpha_t)

        # (|V|, F, 2)
        pred_X = logit_X.softmax(dim=-1)
        # (F, |V|, 2)
        pred_X = torch.transpose(pred_X, 0, 1)

        num_nodes = X_t_one_hot.size(0)
        # (|V|, F, 2)
        X_t_one_hot = X_t_one_hot.reshape(num_nodes, self.num_attrs_X, -1)
        # (F, |V|, 2)
        X_t_one_hot = torch.transpose(X_t_one_hot, 0, 1)

        return self.denoise_match_Z(X_t_one_hot,
                                    Q_t_X,
                                    X_one_hot_3d,
                                    Q_bar_s_X,
                                    pred_X)

    @torch.no_grad()
    def val_step(self,
                 X_one_hot_3d,
                 E_one_hot,
                 Y,
                 batch_src,
                 batch_dst,
                 batch_E_one_hot):
        """Perform a validation step.

        Parameters
        ----------
        X_one_hot_3d : torch.Tensor of shape (F, |V|, 2)
            X_one_hot_3d[f, :, :] is the one-hot encoding of the f-th node attribute
            in the real graph.
        E_one_hot : torch.Tensor of shape (|V|, |V|, 2)
            - E_one_hot[:, :, 0] indicates the absence of an edge in the real graph.
            - E_one_hot[:, :, 1] is the adjacency matrix of the real graph.
        Y : torch.Tensor of shape (|V|)
            Categorical node labels of the real graph.
        batch_src : torch.LongTensor of shape (B)
            Source node IDs for a batch of candidate edges (node pairs).
        batch_dst : torch.LongTensor of shape (B)
            Destination node IDs for a batch of candidate edges (node pairs).
        batch_E_one_hot : torch.Tensor of shape (B, 2)
            E_one_hot[batch_dst, batch_src].

        Returns
        -------
        denoise_match_E : float
            Denoising matching term for edge.
        denoise_match_X : float
            Denoising matching term for node attributes.
        log_p_0_E : float
            Reconstruction term for edge.
        log_p_0_X : float
            Reconstruction term for node attributes.
        """
        device = E_one_hot.device

        denoise_match_X = []
        denoise_match_E = []

        # t=0 is handled separately.
        for t_sample in range(1, self.T + 1):
            t = torch.LongTensor([t_sample]).to(device)
            t_float, X_t_one_hot, E_t = self.apply_noise(
                X_one_hot_3d, E_one_hot, t)
            A_t = self.get_adj(E_t)
            logit_X, logit_E = self.graph_encoder(t_float,
                                                  X_t_one_hot,
                                                  Y,
                                                  A_t,
                                                  batch_src,
                                                  batch_dst)

            E_t_one_hot = F.one_hot(E_t[batch_src, batch_dst],
                                    num_classes=self.num_classes_E).float()
            denoise_match_E_t = self.denoise_match_E(t_float,
                                                     logit_E,
                                                     E_t_one_hot,
                                                     batch_E_one_hot)
            denoise_match_E.append(denoise_match_E_t)

            denoise_match_X_t = self.denoise_match_X(t_float,
                                                     logit_X,
                                                     X_t_one_hot,
                                                     X_one_hot_3d)
            denoise_match_X.append(denoise_match_X_t)

        denoise_match_E = float(np.mean(denoise_match_E)) * self.T
        denoise_match_X = float(np.mean(denoise_match_X)) * self.T

        # t=0
        t_0 = torch.LongTensor([0]).to(device)
        loss_X, loss_E = self.log_p_t(X_one_hot_3d,
                                      E_one_hot,
                                      Y,
                                      batch_src,
                                      batch_dst,
                                      batch_E_one_hot,
                                      t_0)
        log_p_0_E = loss_E.item()
        log_p_0_X = loss_X.item()

        return denoise_match_E, denoise_match_X,\
            log_p_0_E, log_p_0_X

    @torch.no_grad()
    def sample(self, batch_size=32768, num_workers=4):
        """Sample a graph.

        Parameters
        ----------
        batch_size : int
            Batch size for edge prediction.
        num_workers : int
            Number of subprocesses for data loading in edge prediction.

        Returns
        -------
        X_t_one_hot : torch.Tensor of shape (F, |V|, 2)
            One-hot encoding of the generated node attributes.
        Y_0_one_hot : torch.Tensor of shape (|V|, C)
            One-hot encoding of the generated node labels.
        E_t : torch.LongTensor of shape (|V|, |V|)
            Adjacency matrix of the generated graph.
        """
        device = self.X_marginal.device
        dst, src = torch.triu_indices(self.num_nodes, self.num_nodes,
                                      offset=1, device=device)
        # (|E|)
        self.dst = dst
        # (|E|)
        self.src = src
        # (|E|, 2)
        edge_index = torch.stack([dst, src], dim=1).to("cpu")
        data_loader = DataLoader(edge_index, batch_size=batch_size,
                                 num_workers=num_workers)

        # Sample G^T from prior distribution.
        # (|V|, C)
        Y_prior = self.Y_marginal[None, :].expand(self.num_nodes, -1)
        # (|V|)
        Y_0 = Y_prior.multinomial(1).reshape(-1)

        # (|V|, |V|, 2)
        E_prior = self.E_marginal[None, None, :].expand(
            self.num_nodes, self.num_nodes, -1)
        # (|V|, |V|)
        E_t = self.sample_E(E_prior)

        # (F, |V|, 2)
        X_prior = self.X_marginal[:, None, :].expand(-1, self.num_nodes, -1)
        # (|V|, 2F)
        X_t_one_hot = self.sample_X(X_prior)

        # Iteratively sample p(D^s | D^t) for t = 1, ..., T, with s = t - 1.
        for s in tqdm(list(reversed(range(0, self.T)))):
            t = s + 1

            # Note that computing Q_bar_t from alpha_bar_t is the same
            # as computing Q_t from alpha_t.
            alpha_t = self.noise_schedule.alphas[t]
            alpha_bar_s = self.noise_schedule.alpha_bars[s]
            alpha_bar_t = self.noise_schedule.alpha_bars[t]

            Q_t_E = self.transition.get_Q_bar_E(alpha_t)
            Q_bar_s_E = self.transition.get_Q_bar_E(alpha_bar_s)
            Q_bar_t_E = self.transition.get_Q_bar_E(alpha_bar_t)

            t_float = torch.tensor([t / self.T]).to(device)

            A_t, E_s = self.get_E_t(device,
                                    data_loader,
                                    self.graph_encoder.pred_E,
                                    t_float,
                                    X_t_one_hot,
                                    Y_0,
                                    E_t,
                                    Q_t_E,
                                    Q_bar_s_E,
                                    Q_bar_t_E,
                                    batch_size)

            # (|V|, F, 2)
            pred_X = self.graph_encoder.pred_X(t_float,
                                               X_t_one_hot,
                                               Y_0,
                                               A_t)
            pred_X = pred_X.softmax(dim=-1)
            # (F, |V|, 2)
            pred_X = torch.transpose(pred_X, 0, 1)

            # (|V|, F, 2)
            X_t_one_hot = X_t_one_hot.reshape(self.num_nodes, self.num_attrs_X, -1)
            # (F, |V|, 2)
            X_t_one_hot = torch.transpose(X_t_one_hot, 0, 1)

            # (F, |V|, 2)
            Q_t_X = self.transition.get_Q_bar_X(alpha_t)
            Q_bar_s_X = self.transition.get_Q_bar_X(alpha_bar_s)
            Q_bar_t_X = self.transition.get_Q_bar_X(alpha_bar_t)

            X_prob = self.posterior(X_t_one_hot, Q_t_X,
                                    Q_bar_s_X, Q_bar_t_X, pred_X)
            X_t_one_hot = self.sample_X(X_prob)
            E_t = E_s

        # (|V|, F, 2)
        X_t_one_hot = X_t_one_hot.reshape(self.num_nodes, self.num_attrs_X, -1)
        # (F, |V|, 2)
        X_t_one_hot = torch.transpose(X_t_one_hot, 0, 1)

        Y_0_one_hot = F.one_hot(Y_0, num_classes=self.num_classes_Y).float()

        return X_t_one_hot, Y_0_one_hot, E_t

class ModelAsync(BaseModel):
    """
    Parameters
    ----------
    T_X : int
        Number of diffusion time steps - 1 for node attributes.
    T_E : int
        Number of diffusion time steps - 1 for edges.
    X_marginal : torch.Tensor of shape (F, 2)
        X_marginal[f, :] is the marginal distribution of the f-th node attribute.
    Y_marginal : torch.Tensor of shape (C)
        Marginal distribution of the node labels.
    E_marginal : torch.Tensor of shape (2)
        Marginal distribution of the edge existence.
    mlp_X_config : dict
        Configuration of the MLP for reconstructing node attributes.
    gnn_E_config : dict
        Configuration of the GNN for reconstructing edges.
    num_nodes : int
        Number of nodes in the original graph.
    """
    def __init__(self,
                 T_X,
                 T_E,
                 X_marginal,
                 Y_marginal,
                 E_marginal,
                 mlp_X_config,
                 gnn_E_config,
                 num_nodes):
        super().__init__(T=T_X,
                         X_marginal=X_marginal,
                         Y_marginal=Y_marginal,
                         E_marginal=E_marginal,
                         num_nodes=num_nodes)

        del self.T
        del self.noise_schedule

        self.T_X = T_X
        self.T_E = T_E

        device = X_marginal.device
        self.noise_schedule_X = NoiseSchedule(T_X, device)
        self.noise_schedule_E = NoiseSchedule(T_E, device)

        self.graph_encoder = GNNAsymm(num_attrs_X=self.num_attrs_X,
                                      num_classes_X=self.num_classes_X,
                                      num_classes_Y=self.num_classes_Y,
                                      num_classes_E=self.num_classes_E,
                                      mlp_X_config=mlp_X_config,
                                      gnn_E_config=gnn_E_config)

        self.loss_X = LossX(self.num_attrs_X, self.num_classes_X)

    def apply_noise_X(self, X_one_hot_3d, t=None):
        """Corrupt X and sample X^t.

        Parameters
        ----------
        X_one_hot_3d : torch.Tensor of shape (F, |V|, 2)
            X_one_hot_3d[f, :, :] is the one-hot encoding of the f-th node attribute
            in the real graph.
        t : torch.LongTensor of shape (1), optional
            If specified, a time step will be enforced rather than sampled.

        Returns
        -------
        t_float_X : torch.Tensor of shape (1)
            Sampled timestep divided by self.T_X.
        X_t_one_hot : torch.Tensor of shape (|V|, 2 * F)
            One-hot encoding of the sampled node attributes.
        """
        if t is None:
            # Sample a timestep t uniformly.
            # Note that the notation is slightly inconsistent with the paper.
            # t=0 corresponds to t=1 in the paper, where corruption has already taken place.
            t = torch.randint(low=0, high=self.T_X + 1, size=(1,),
                              device=X_one_hot_3d.device)

        alpha_bar_t = self.noise_schedule_X.alpha_bars[t]

        Q_bar_t_X = self.transition.get_Q_bar_X(alpha_bar_t) # (F, 2, 2)
        # Compute matrix multiplication over the first batch dimension.
        prob_X = torch.bmm(X_one_hot_3d, Q_bar_t_X)          # (F, |V|, 2)

        # Sample X_t.
        X_t_one_hot = self.sample_X(prob_X)

        t_float_X = t / self.T_X

        return t_float_X, X_t_one_hot

    def apply_noise_E(self, E_one_hot, t=None):
        """Corrupt A and sample A^t.

        Parameters
        ----------
        E_one_hot : torch.Tensor of shape (|V|, |V|, 2)
            - E_one_hot[:, :, 0] indicates the absence of an edge in the real graph.
            - E_one_hot[:, :, 1] is the adjacency matrix of the real graph.
        t : torch.LongTensor of shape (1), optional
            If specified, a time step will be enforced rather than sampled.

        Returns
        -------
        t_float_E : torch.Tensor of shape (1)
            Sampled timestep divided by self.T_E.
        E_t : torch.LongTensor of shape (|V|, |V|)
            Sampled symmetric adjacency matrix.
        """
        if t is None:
            # Sample a timestep t uniformly.
            # Note that the notation is slightly inconsistent with the paper.
            # t=0 corresponds to t=1 in the paper, where corruption has already taken place.
            t = torch.randint(low=0, high=self.T_E + 1, size=(1,),
                              device=E_one_hot.device)

        alpha_bar_t = self.noise_schedule_E.alpha_bars[t]

        Q_bar_t_E = self.transition.get_Q_bar_E(alpha_bar_t) # (2, 2)
        prob_E = E_one_hot @ Q_bar_t_E                       # (|V|, |V|, 2)
        E_t = self.sample_E(prob_E)

        t_float_E = t / self.T_E

        return t_float_E, E_t

    def log_p_t(self,
                X_one_hot_3d,
                E_one_hot,
                Y,
                X_one_hot_2d,
                batch_src,
                batch_dst,
                batch_E_one_hot,
                t_X=None,
                t_E=None):
        """Obtain G^t and compute log p(G | G^t, Y, t).

        Parameters
        ----------
        X_one_hot_3d : torch.Tensor of shape (F, |V|, 2)
            X_one_hot_3d[f, :, :] is the one-hot encoding of the f-th node attribute
            in the real graph.
        E_one_hot : torch.Tensor of shape (|V|, |V|, 2)
            - E_one_hot[:, :, 0] indicates the absence of an edge in the real graph.
            - E_one_hot[:, :, 1] is the adjacency matrix of the real graph.
        Y : torch.Tensor of shape (|V|)
            Categorical node labels of the real graph.
        X_one_hot_2d : torch.Tensor of shape (|V|, 2 * F)
            Flattened one-hot encoding of the node attributes in the real graph.
        batch_src : torch.LongTensor of shape (B)
            Source node IDs for a batch of candidate edges (node pairs).
        batch_dst : torch.LongTensor of shape (B)
            Destination node IDs for a batch of candidate edges (node pairs).
        batch_E_one_hot : torch.Tensor of shape (B, 2)
            E_one_hot[batch_dst, batch_src].
        t_X : torch.LongTensor of shape (1), optional
            If specified, a time step will be enforced rather than sampled for
            node attributes.
        t_E : torch.LongTensor of shape (1), optional
            If specified, a time step will be enforced rather than sampled for
            edges.

        Returns
        -------
        loss_X : torch.Tensor
            Scalar representing the loss for node attributes.
        loss_E : torch.Tensor
            Scalar representing the loss for edge existence.
        """
        t_float_X, X_t_one_hot = self.apply_noise_X(X_one_hot_3d, t_X)
        t_float_E, E_t = self.apply_noise_E(E_one_hot, t_E)
        A_t = self.get_adj(E_t)
        logit_X, logit_E = self.graph_encoder(t_float_X,
                                              t_float_E,
                                              X_t_one_hot,
                                              Y,
                                              X_one_hot_2d,
                                              A_t,
                                              batch_src,
                                              batch_dst)

        loss_X = self.loss_X(X_one_hot_3d, logit_X)
        loss_E = self.loss_E(batch_E_one_hot, logit_E)

        return loss_X, loss_E

    def denoise_match_X(self,
                        t_float,
                        logit_X,
                        X_t_one_hot,
                        X_one_hot_3d):
        """Compute the denoising match term for node attribute prediction given a
        sampled t, i.e., the KL divergence between q(D^{t-1}| D, D^t) and
        q(D^{t-1}| hat{p}^{D}, D^t).

        Parameters
        ----------
        t_float : torch.Tensor of shape (1)
            Sampled timestep divided by self.T_X.
        logit_X : torch.Tensor of shape (|V|, F, 2)
            Predicted logits for the node attributes.
        X_t_one_hot : torch.Tensor of shape (|V|, 2 * F)
            One-hot encoding of the node attributes sampled at time step t.
        X_one_hot_3d : torch.Tensor of shape (F, |V|, 2)
            X_one_hot_3d[f, :, :] is the one-hot encoding of the f-th node attribute.

        Returns
        -------
        float
            KL value for node attributes.
        """
        t = int(t_float.item() * self.T_X)
        s = t - 1

        alpha_bar_s = self.noise_schedule_X.alpha_bars[s]
        alpha_t = self.noise_schedule_X.alphas[t]

        Q_bar_s_X = self.transition.get_Q_bar_X(alpha_bar_s)
        # Note that computing Q_bar_t from alpha_bar_t is the same
        # as computing Q_t from alpha_t.
        Q_t_X = self.transition.get_Q_bar_X(alpha_t)

        # (|V|, F, 2)
        pred_X = logit_X.softmax(dim=-1)
        # (F, |V|, 2)
        pred_X = torch.transpose(pred_X, 0, 1)

        num_nodes = X_t_one_hot.size(0)
        # (|V|, F, 2)
        X_t_one_hot = X_t_one_hot.reshape(num_nodes, self.num_attrs_X, -1)
        # (F, |V|, 2)
        X_t_one_hot = torch.transpose(X_t_one_hot, 0, 1)

        return self.denoise_match_Z(X_t_one_hot,
                                    Q_t_X,
                                    X_one_hot_3d,
                                    Q_bar_s_X,
                                    pred_X)

    def denoise_match_E(self,
                        t_float,
                        logit_E,
                        E_t_one_hot,
                        E_one_hot):
        """Compute the denoising match term for edge prediction given a
        sampled t, i.e., the KL divergence between q(D^{t-1}| D, D^t) and
        q(D^{t-1}| hat{p}^{D}, D^t).

        Parameters
        ----------
        t_float : torch.Tensor of shape (1)
            Sampled timestep divided by self.T_E.
        logit_E : torch.Tensor of shape (B, 2)
            Predicted logits for the edge existence of a batch of node pairs.
        E_t_one_hot : torch.Tensor of shape (B, 2)
            One-hot encoding of sampled edge existence for the batch of
            node pairs.
        E_one_hot : torch.Tensor of shape (B, 2)
            One-hot encoding of the original edge existence for the batch of
            node pairs.

        Returns
        -------
        float
            KL value for edges.
        """
        t = int(t_float.item() * self.T_E)
        s = t - 1

        alpha_bar_s = self.noise_schedule_E.alpha_bars[s]
        alpha_t = self.noise_schedule_E.alphas[t]

        Q_bar_s_E = self.transition.get_Q_bar_E(alpha_bar_s)
        # Note that computing Q_bar_t from alpha_bar_t is the same
        # as computing Q_t from alpha_t.
        Q_t_E = self.transition.get_Q_bar_E(alpha_t)

        pred_E = logit_E.softmax(dim=-1)

        return self.denoise_match_Z(E_t_one_hot,
                                    Q_t_E,
                                    E_one_hot,
                                    Q_bar_s_E,
                                    pred_E)

    @torch.no_grad()
    def val_step(self,
                 X_one_hot_3d,
                 E_one_hot,
                 Y,
                 X_one_hot_2d,
                 batch_src,
                 batch_dst,
                 batch_E_one_hot):
        """Perform a validation step.

        Parameters
        ----------
        X_one_hot_3d : torch.Tensor of shape (F, |V|, 2)
            X_one_hot_3d[f, :, :] is the one-hot encoding of the f-th node attribute
            in the real graph.
        E_one_hot : torch.Tensor of shape (|V|, |V|, 2)
            - E_one_hot[:, :, 0] indicates the absence of an edge in the real graph.
            - E_one_hot[:, :, 1] is the adjacency matrix of the real graph.
        Y : torch.Tensor of shape (|V|)
            Categorical node labels of the real graph.
        X_one_hot_2d : torch.Tensor of shape (|V|, 2 * F)
            Flattened one-hot encoding of the node attributes in the real graph.
        batch_src : torch.LongTensor of shape (B)
            Source node IDs for a batch of candidate edges (node pairs).
        batch_dst : torch.LongTensor of shape (B)
            Destination node IDs for a batch of candidate edges (node pairs).
        batch_E_one_hot : torch.Tensor of shape (B, 2)
            E_one_hot[batch_dst, batch_src].

        Returns
        -------
        denoise_match_E : float
            Denoising matching term for edge.
        denoise_match_X : float
            Denoising matching term for node attributes.
        log_p_0_E : float
            Reconstruction term for edge.
        log_p_0_X : float
            Reconstruction term for node attributes.
        """
        device = E_one_hot.device

        # Case1: X
        denoise_match_X = []
        # t=0 is handled separately.
        for t_sample_X in range(1, self.T_X + 1):
            t_X = torch.LongTensor([t_sample_X]).to(device)
            t_float_X, X_t_one_hot = self.apply_noise_X(X_one_hot_3d, t_X)
            logit_X = self.graph_encoder.pred_X(t_float_X,
                                                X_t_one_hot,
                                                Y)

            denoise_match_X_t = self.denoise_match_X(t_float_X,
                                                     logit_X,
                                                     X_t_one_hot,
                                                     X_one_hot_3d)
            denoise_match_X.append(denoise_match_X_t)
        denoise_match_X = float(np.mean(denoise_match_X)) * self.T_X

        # Case2: E
        denoise_match_E = []
        # t=0 is handled separately.
        for t_sample_E in range(1, self.T_E + 1):
            t_E = torch.LongTensor([t_sample_E]).to(device)
            t_float_E, E_t = self.apply_noise_E(E_one_hot, t_E)
            A_t = self.get_adj(E_t)
            logit_E = self.graph_encoder.pred_E(t_float_E,
                                                X_one_hot_2d,
                                                Y,
                                                A_t,
                                                batch_src,
                                                batch_dst)

            E_t_one_hot = F.one_hot(E_t[batch_src, batch_dst],
                                    num_classes=self.num_classes_E).float()
            denoise_match_E_t = self.denoise_match_E(t_float_E,
                                                     logit_E,
                                                     E_t_one_hot,
                                                     batch_E_one_hot)
            denoise_match_E.append(denoise_match_E_t)
        denoise_match_E = float(np.mean(denoise_match_E)) * self.T_E

        # t=0
        t_0 = torch.LongTensor([0]).to(device)
        loss_X, loss_E = self.log_p_t(X_one_hot_3d,
                                      E_one_hot,
                                      Y,
                                      X_one_hot_2d,
                                      batch_src,
                                      batch_dst,
                                      batch_E_one_hot,
                                      t_X=t_0,
                                      t_E=t_0)
        log_p_0_E = loss_E.item()
        log_p_0_X = loss_X.item()

        return denoise_match_E, denoise_match_X,\
            log_p_0_E, log_p_0_X

    @torch.no_grad()
    def sample(self, batch_size=32768, num_workers=4):
        """Sample a graph.

        Parameters
        ----------
        batch_size : int
            Batch size for edge prediction.
        num_workers : int
            Number of subprocesses for data loading in edge prediction.

        Returns
        -------
        X_t_one_hot : torch.Tensor of shape (F, |V|, 2)
            One-hot encoding of the generated node attributes.
        Y_0_one_hot : torch.Tensor of shape (|V|, C)
            One-hot encoding of the generated node labels.
        E_t : torch.LongTensor of shape (|V|, |V|)
            Adjacency matrix of the generated graph.
        """
        device = self.X_marginal.device

        # Sample Y_0
        # (|V|, C)
        Y_prior = self.Y_marginal[None, :].expand(self.num_nodes, -1)
        # (|V|)
        Y_0 = Y_prior.multinomial(1).reshape(-1)

        # Sample X^T from prior distribution.
        # (F, |V|, 2)
        X_prior = self.X_marginal[:, None, :].expand(-1, self.num_nodes, -1)
        # (|V|, 2F)
        X_t_one_hot = self.sample_X(X_prior)

        # Iteratively sample p(X^s | X^t) for t = 1, ..., T_X, with s = t - 1.
        for s_X in tqdm(list(reversed(range(0, self.T_X)))):
            t_X = s_X + 1
            t_float_X = torch.tensor([t_X / self.T_X]).to(device)
            pred_X = self.graph_encoder.pred_X(t_float_X,
                                               X_t_one_hot,
                                               Y_0)
            pred_X = pred_X.softmax(dim=-1)
            # (F, |V|, 2)
            pred_X = torch.transpose(pred_X, 0, 1)

            # (|V|, F, 2)
            X_t_one_hot = X_t_one_hot.reshape(self.num_nodes, self.num_attrs_X, -1)
            # (F, |V|, 2)
            X_t_one_hot = torch.transpose(X_t_one_hot, 0, 1)

            # Note that computing Q_bar_t from alpha_bar_t is the same
            # as computing Q_t from alpha_t.
            alpha_t_X = self.noise_schedule_X.alphas[t_X]
            alpha_bar_s_X = self.noise_schedule_X.alpha_bars[s_X]
            alpha_bar_t_X = self.noise_schedule_X.alpha_bars[t_X]

            # (F, |V|, 2)
            Q_t_X = self.transition.get_Q_bar_X(alpha_t_X)
            Q_bar_s_X = self.transition.get_Q_bar_X(alpha_bar_s_X)
            Q_bar_t_X = self.transition.get_Q_bar_X(alpha_bar_t_X)

            X_prob = self.posterior(X_t_one_hot, Q_t_X,
                                    Q_bar_s_X, Q_bar_t_X, pred_X)
            X_t_one_hot = self.sample_X(X_prob)

        # Sample E^T from prior distribution.
        # (|V|, |V|, 2)
        E_prior = self.E_marginal[None, None, :].expand(
            self.num_nodes, self.num_nodes, -1)
        # (|V|, |V|)
        E_t = self.sample_E(E_prior)

        dst, src = torch.triu_indices(self.num_nodes, self.num_nodes,
                                      offset=1, device=device)
        # (|E|)
        self.dst = dst
        # (|E|)
        self.src = src
        # (|E|, 2)
        edge_index = torch.stack([dst, src], dim=1).to("cpu")
        data_loader = DataLoader(edge_index, batch_size=batch_size,
                                 num_workers=num_workers)

        # Iteratively sample p(A^s | A^t) for t = 1, ..., T_E, with s = t - 1.
        for s_E in tqdm(list(reversed(range(0, self.T_E)))):
            t_E = s_E + 1
            alpha_t_E = self.noise_schedule_E.alphas[t_E]
            alpha_bar_s_E = self.noise_schedule_E.alpha_bars[s_E]
            alpha_bar_t_E = self.noise_schedule_E.alpha_bars[t_E]

            Q_t_E = self.transition.get_Q_bar_E(alpha_t_E)
            Q_bar_s_E = self.transition.get_Q_bar_E(alpha_bar_s_E)
            Q_bar_t_E = self.transition.get_Q_bar_E(alpha_bar_t_E)

            t_float_E = torch.tensor([t_E / self.T_E]).to(device)

            _, E_t = self.get_E_t(device,
                                  data_loader,
                                  self.graph_encoder.pred_E,
                                  t_float_E,
                                  X_t_one_hot,
                                  Y_0,
                                  E_t,
                                  Q_t_E,
                                  Q_bar_s_E,
                                  Q_bar_t_E,
                                  batch_size)

        # (|V|, F, 2)
        X_t_one_hot = X_t_one_hot.reshape(self.num_nodes, self.num_attrs_X, -1)
        # (F, |V|, 2)
        X_t_one_hot = torch.transpose(X_t_one_hot, 0, 1)

        Y_0_one_hot = F.one_hot(Y_0, num_classes=self.num_classes_Y).float()

        return X_t_one_hot, Y_0_one_hot, E_t
