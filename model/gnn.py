import torch
import torch.nn as nn

__all__ = ["GNN", "LinkPredictor", "GNNAsymm"]

class GNNLayer(nn.Module):
    """Graph Neural Network (GNN) / Message Passing Neural Network (MPNN) Layer.

    Parameters
    ----------
    hidden_X : int
        Hidden size for the node attributes.
    hidden_Y : int
        Hidden size for the node label.
    hidden_t : int
        Hidden size for the normalized time step.
    dropout : float
        Dropout rate.
    """
    def __init__(self,
                 hidden_X,
                 hidden_Y,
                 hidden_t,
                 dropout):
        super().__init__()

        self.update_X = nn.Sequential(
            nn.Linear(hidden_X + hidden_Y + hidden_t, hidden_X),
            nn.ReLU(),
            nn.LayerNorm(hidden_X),
            nn.Dropout(dropout)
        )
        self.update_Y = nn.Sequential(
            nn.Linear(hidden_Y, hidden_Y),
            nn.ReLU(),
            nn.LayerNorm(hidden_Y),
            nn.Dropout(dropout)
        )

    def forward(self, A, h_X, h_Y, h_t):
        """
        Parameters
        ----------
        A : dglsp.SparseMatrix
            Adjacency matrix.
        h_X : torch.Tensor of shape (|V|, hidden_X)
            Hidden representations for the node attributes.
        h_Y : torch.Tensor of shape (|V|, hidden_Y)
            Hidden representations for the node label.
        h_t : torch.Tensor of shape (|V|, hidden_t)
            Hidden representations for the normalized time step.

        Returns
        -------
        h_X : torch.Tensor of shape (|V|, hidden_X)
            Updated hidden representations for the node attributes.
        h_Y : torch.Tensor of shape (|V|, hidden_Y)
            Updated hidden representations for the node label.
        """
        h_aggr_X = A @ torch.cat([h_X, h_Y], dim=1)
        h_aggr_Y = A @ h_Y

        num_nodes = h_X.size(0)
        h_t_expand = h_t.expand(num_nodes, -1)
        h_aggr_X = torch.cat([h_aggr_X, h_t_expand], dim=1)

        h_X = self.update_X(h_aggr_X)
        h_Y = self.update_Y(h_aggr_Y)

        return h_X, h_Y

class GNNTower(nn.Module):
    """Graph Neural Network (GNN) / Message Passing Neural Network (MPNN).

    Parameters
    ----------
    num_attrs_X : int
        Number of node attributes.
    num_classes_X : int
        Number of classes for each node attribute.
    num_classes_Y : int
        Number of classes for node label.
    hidden_t : int
        Hidden size for the normalized time step.
    hidden_X : int
        Hidden size for the node attributes.
    hidden_Y : int
        Hidden size for the node label.
    out_size : int
        Output size of the final MLP layer.
    num_gnn_layers : int
        Number of GNN/MPNN layers.
    dropout : float
        Dropout rate.
    node_mode : bool
        Whether the encoder is used for node attribute prediction or structure
        prediction.
    """
    def __init__(self,
                 num_attrs_X,
                 num_classes_X,
                 num_classes_Y,
                 hidden_t,
                 hidden_X,
                 hidden_Y,
                 out_size,
                 num_gnn_layers,
                 dropout,
                 node_mode):
        super().__init__()

        in_X = num_attrs_X * num_classes_X
        self.num_attrs_X = num_attrs_X
        self.num_classes_X = num_classes_X

        self.mlp_in_t = nn.Sequential(
            nn.Linear(1, hidden_t),
            nn.ReLU(),
            nn.Linear(hidden_t, hidden_t),
            nn.ReLU())
        self.mlp_in_X = nn.Sequential(
            nn.Linear(in_X, hidden_X),
            nn.ReLU(),
            nn.Linear(hidden_X, hidden_X),
            nn.ReLU()
        )
        self.emb_Y = nn.Embedding(num_classes_Y, hidden_Y)

        self.gnn_layers = nn.ModuleList([
            GNNLayer(hidden_X,
                     hidden_Y,
                     hidden_t,
                     dropout)
            for _ in range(num_gnn_layers)])

        # +1 for the input attributes
        hidden_cat = (num_gnn_layers + 1) * (hidden_X + hidden_Y) + hidden_t
        self.mlp_out = nn.Sequential(
            nn.Linear(hidden_cat, hidden_cat),
            nn.ReLU(),
            nn.Linear(hidden_cat, out_size)
        )

        self.node_mode = node_mode

    def forward(self,
                t_float,
                X_t_one_hot,
                Y_real,
                A_t):
        # Input projection.
        # (1, hidden_t)
        h_t = self.mlp_in_t(t_float).unsqueeze(0)
        h_X = self.mlp_in_X(X_t_one_hot)
        h_Y = self.emb_Y(Y_real)

        h_X_list = [h_X]
        h_Y_list = [h_Y]
        for gnn in self.gnn_layers:
            h_X, h_Y = gnn(A_t, h_X, h_Y, h_t)
            h_X_list.append(h_X)
            h_Y_list.append(h_Y)

        # (|V|, hidden_t)
        h_t = h_t.expand(h_X.size(0), -1)
        h_cat = torch.cat(h_X_list + h_Y_list + [h_t], dim=1)

        if self.node_mode:
            # (|V|, F * C_X)
            logit = self.mlp_out(h_cat)
            # (|V|, F, C_X)
            logit = logit.reshape(Y_real.size(0), self.num_attrs_X, -1)

            return logit
        else:
            return self.mlp_out(h_cat)

class LinkPredictor(nn.Module):
    """Model for structure prediction.

    Parameters
    ----------
    num_attrs_X : int
        Number of node attributes.
    num_classes_X : int
        Number of classes for each node attribute.
    num_classes_Y : int
        Number of classes for node label.
    num_classes_E : int
        Number of edge classes.
    hidden_t : int
        Hidden size for the normalized time step.
    hidden_X : int
        Hidden size for the node attributes.
    hidden_Y : int
        Hidden size for the node label.
    hidden_E : int
        Hidden size for the edges.
    num_gnn_layers : int
        Number of GNN/MPNN layers.
    dropout : float
        Dropout rate.
    """
    def __init__(self,
                 num_attrs_X,
                 num_classes_X,
                 num_classes_Y,
                 num_classes_E,
                 hidden_t,
                 hidden_X,
                 hidden_Y,
                 hidden_E,
                 num_gnn_layers,
                 dropout):
        super().__init__()

        self.gnn_encoder = GNNTower(num_attrs_X,
                                    num_classes_X,
                                    num_classes_Y,
                                    hidden_t,
                                    hidden_X,
                                    hidden_Y,
                                    hidden_E,
                                    num_gnn_layers,
                                    dropout,
                                    node_mode=False)
        self.mlp_out = nn.Sequential(
            nn.Linear(hidden_E, hidden_E),
            nn.ReLU(),
            nn.Linear(hidden_E, num_classes_E)
        )

    def forward(self,
                t_float,
                X_t_one_hot,
                Y_real,
                A_t,
                src,
                dst):
        # (|V|, hidden_E)
        h = self.gnn_encoder(t_float,
                             X_t_one_hot,
                             Y_real,
                             A_t)
        # (|E|, hidden_E)
        h = h[src] * h[dst]
        # (|E|, num_classes_E)
        logit = self.mlp_out(h)

        return logit

class GNN(nn.Module):
    """P(X|Y, X^t, A^t) + P(A|Y, X^t, A^t)

    Parameters
    ----------
    num_attrs_X : int
        Number of node attributes.
    num_classes_X : int
        Number of classes for each node attribute.
    num_classes_Y : int
        Number of classes for node label.
    num_classes_E : int
        Number of edge classes.
    gnn_X_config : dict
        Configuration of the GNN for reconstructing node attributes.
    gnn_E_config : dict
        Configuration of the GNN for reconstructing edges.
    """
    def __init__(self,
                 num_attrs_X,
                 num_classes_X,
                 num_classes_Y,
                 num_classes_E,
                 gnn_X_config,
                 gnn_E_config):
        super().__init__()

        self.pred_X = GNNTower(num_attrs_X,
                               num_classes_X,
                               num_classes_Y,
                               out_size=num_attrs_X * num_classes_X,
                               node_mode=True,
                               **gnn_X_config)

        self.pred_E = LinkPredictor(num_attrs_X,
                                    num_classes_X,
                                    num_classes_Y,
                                    num_classes_E,
                                    **gnn_E_config)

    def forward(self,
                t_float,
                X_t_one_hot,
                Y,
                A_t,
                batch_src,
                batch_dst):
        """
        Parameters
        ----------
        t_float : torch.Tensor of shape (1)
            Sampled timestep divided by self.T.
        X_t_one_hot : torch.Tensor of shape (|V|, 2 * F)
            One-hot encoding of the sampled node attributes.
        Y : torch.Tensor of shape (|V|)
            Categorical node labels.
        A_t : dglsp.SparseMatrix
            Row-normalized sampled adjacency matrix.
        batch_src : torch.LongTensor of shape (B)
            Source node IDs for a batch of candidate edges (node pairs).
        batch_dst : torch.LongTensor of shape (B)
            Destination node IDs for a batch of candidate edges (node pairs).

        Returns
        -------
        logit_X : torch.Tensor of shape (|V|, F, 2)
            Predicted logits for the node attributes.
        logit_E : torch.Tensor of shape (B, 2)
            Predicted logits for the edge existence.
        """
        logit_X = self.pred_X(t_float,
                              X_t_one_hot,
                              Y,
                              A_t)

        logit_E = self.pred_E(t_float,
                              X_t_one_hot,
                              Y,
                              A_t,
                              batch_src,
                              batch_dst)

        return logit_X, logit_E

class MLPLayer(nn.Module):
    """
    Parameters
    ----------
    hidden_X : int
        Hidden size for the node attributes.
    hidden_Y : int
        Hidden size for the node labels.
    hidden_t : int
        Hidden size for the normalized time step.
    dropout : float
        Dropout rate.
    """
    def __init__(self,
                 hidden_X,
                 hidden_Y,
                 hidden_t,
                 dropout):
        super().__init__()

        self.update_X = nn.Sequential(
            nn.Linear(hidden_X + hidden_Y + hidden_t, hidden_X),
            nn.ReLU(),
            nn.LayerNorm(hidden_X),
            nn.Dropout(dropout)
        )
        self.update_Y = nn.Sequential(
            nn.Linear(hidden_Y, hidden_Y),
            nn.ReLU(),
            nn.LayerNorm(hidden_Y),
            nn.Dropout(dropout)
        )

    def forward(self, h_X, h_Y, h_t):
        """
        Parameters
        ----------
        h_X : torch.Tensor of shape (|V|, hidden_X)
            Hidden representations for the node attributes.
        h_Y : torch.Tensor of shape (|V|, hidden_Y)
            Hidden representations for the node labels.
        h_t : torch.Tensor of shape (1, hidden_t)
            Hidden representations for the normalized time step.

        Returns
        -------
        h_X : torch.Tensor of shape (|V|, hidden_X)
            Updated hidden representations for the node attributes.
        h_Y : torch.Tensor of shape (|V|, hidden_Y)
            Updated hidden representations for the node labels.
        """
        num_nodes = h_X.size(0)
        h_t_expand = h_t.expand(num_nodes, -1)
        h_X = torch.cat([h_X, h_Y, h_t_expand], dim=1)

        h_X = self.update_X(h_X)
        h_Y = self.update_Y(h_Y)

        return h_X, h_Y

class MLPTower(nn.Module):
    def __init__(self,
                 num_attrs_X,
                 num_classes_X,
                 num_classes_Y,
                 hidden_t,
                 hidden_X,
                 hidden_Y,
                 num_mlp_layers,
                 dropout):
        super().__init__()

        in_X = num_attrs_X * num_classes_X
        self.num_attrs_X = num_attrs_X
        self.num_classes_X = num_classes_X

        self.mlp_in_t = nn.Sequential(
            nn.Linear(1, hidden_t),
            nn.ReLU(),
            nn.Linear(hidden_t, hidden_t),
            nn.ReLU())
        self.mlp_in_X = nn.Sequential(
            nn.Linear(in_X, hidden_X),
            nn.ReLU(),
            nn.Linear(hidden_X, hidden_X),
            nn.ReLU()
        )
        self.emb_Y = nn.Embedding(num_classes_Y, hidden_Y)

        self.mlp_layers = nn.ModuleList([
            MLPLayer(hidden_X,
                     hidden_Y,
                     hidden_t,
                     dropout)
            for _ in range(num_mlp_layers)])

        # +1 for the input features
        hidden_cat = (num_mlp_layers + 1) * (hidden_X + hidden_Y) + hidden_t
        self.mlp_out = nn.Sequential(
            nn.Linear(hidden_cat, hidden_cat),
            nn.ReLU(),
            nn.Linear(hidden_cat, in_X)
        )

    def forward(self,
                t_float,
                X_t_one_hot,
                Y_real):
        # Input projection.
        h_t = self.mlp_in_t(t_float).unsqueeze(0)
        h_X = self.mlp_in_X(X_t_one_hot)
        h_Y = self.emb_Y(Y_real)

        h_X_list = [h_X]
        h_Y_list = [h_Y]
        for mlp in self.mlp_layers:
            h_X, h_Y = mlp(h_X, h_Y, h_t)
            h_X_list.append(h_X)
            h_Y_list.append(h_Y)

        h_t = h_t.expand(h_X.size(0), -1)
        h_cat = torch.cat(h_X_list + h_Y_list + [h_t], dim=1)

        logit = self.mlp_out(h_cat)
        # (|V|, F, C)
        logit = logit.reshape(Y_real.size(0), self.num_attrs_X, -1)

        return logit

class GNNAsymm(nn.Module):
    """P(X|Y, X_t) + P(A|Y, X, A_t)

    Parameters
    ----------
    num_attrs_X : int
        Number of node attributes.
    num_classes_X : int
        Number of classes for each node attribute.
    num_classes_Y : int
        Number of classes for node label.
    num_classes_E : int
        Number of edge classes.
    mlp_X_config : dict
        Configuration of the MLP for reconstructing node attributes.
    gnn_E_config : dict
        Configuration of the GNN for reconstructing edges.
    """
    def __init__(self,
                 num_attrs_X,
                 num_classes_X,
                 num_classes_Y,
                 num_classes_E,
                 mlp_X_config,
                 gnn_E_config):
        super().__init__()

        self.pred_X = MLPTower(num_attrs_X,
                               num_classes_X,
                               num_classes_Y,
                               **mlp_X_config)

        self.pred_E = LinkPredictor(num_attrs_X,
                                    num_classes_X,
                                    num_classes_Y,
                                    num_classes_E,
                                    **gnn_E_config)

    def forward(self,
                t_float_X,
                t_float_E,
                X_t_one_hot,
                Y,
                X_one_hot_2d,
                A_t,
                batch_src,
                batch_dst):
        """
        Parameters
        ----------
        t_float_X : torch.Tensor of shape (1)
            Sampled timestep divided by self.T_X.
        t_float_E : torch.Tensor of shape (1)
            Sampled timestep divided by self.T_E.
        X_t_one_hot : torch.Tensor of shape (|V|, 2 * F)
            One-hot encoding of the sampled node attributes.
        Y : torch.Tensor of shape (|V|)
            Categorical node labels.
        X_one_hot_2d : torch.Tensor of shape (|V|, 2 * F)
            Flattened one-hot encoding of the node attributes.
        A_t : dglsp.SparseMatrix
            Row-normalized sampled adjacency matrix.
        batch_src : torch.LongTensor of shape (B)
            Source node IDs for a batch of candidate edges (node pairs).
        batch_dst : torch.LongTensor of shape (B)
            Destination node IDs for a batch of candidate edges (node pairs).

        Returns
        -------
        logit_X : torch.Tensor of shape (|V|, F, 2)
            Predicted logits for the node attributes.
        logit_E : torch.Tensor of shape (B, 2)
            Predicted logits for the edge existence.
        """
        logit_X = self.pred_X(t_float_X,
                              X_t_one_hot,
                              Y)

        logit_E = self.pred_E(t_float_E,
                              X_one_hot_2d,
                              Y,
                              A_t,
                              batch_src,
                              batch_dst)

        return logit_X, logit_E
