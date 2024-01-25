import dgl
import torch
import torch.nn.functional as F

from dgl.data import AmazonCoBuyPhotoDataset, AmazonCoBuyComputerDataset, \
    CoraGraphDataset

def load_dataset(data_name):
    if data_name == "cora":
        dataset = CoraGraphDataset()
    elif data_name == "amazon_photo":
        dataset = AmazonCoBuyPhotoDataset()
    elif data_name == "amazon_computer":
        dataset = AmazonCoBuyComputerDataset()

    g = dataset[0]
    g = dgl.remove_self_loop(g)

    X = g.ndata['feat']
    X[X != 0] = 1.

    # Remove columns with constant values.
    non_full_zero_feat_mask = X.sum(dim=0) != 0
    X = X[:, non_full_zero_feat_mask]

    non_full_one_feat_mask = X.sum(dim=0) != X.size(0)
    X = X[:, non_full_one_feat_mask]

    g.ndata['feat'] = X
    return g

def preprocess(g):
    """Prepare data for GraphMaker.

    Parameters
    ----------
    g : DGLGraph
        Graph to be preprocessed.

    Returns
    -------
    X_one_hot : torch.Tensor of shape (F, N, 2)
        X_one_hot[f, :, :] is the one-hot encoding of the f-th node attribute.
        N = |V|.
    Y : torch.Tensor of shape (N)
        Categorical node labels.
    E_one_hot : torch.Tensor of shape (N, N, 2)
        - E_one_hot[:, :, 0] indicates the absence of an edge.
        - E_one_hot[:, :, 1] is the original adjacency matrix.
    X_marginal : torch.Tensor of shape (F, 2)
        X_marginal[f, :] is the marginal distribution of the f-th node attribute.
    Y_marginal : torch.Tensor of shape (C)
        Marginal distribution of the node labels.
    E_marginal : torch.Tensor of shape (2)
        Marginal distribution of the edge existence.
    X_cond_Y_marginals : torch.Tensor of shape (F, C, 2)
        X_cond_Y_marginals[f, k] is the marginal distribution of the f-th node
        attribute conditioned on the node label being k.
    """
    X = g.ndata['feat']
    Y = g.ndata['label']
    N = g.num_nodes()
    src, dst = g.edges()

    X_one_hot_list = []
    for f in range(X.size(1)):
        # (N, 2)
        X_f_one_hot = F.one_hot(X[:, f].long())
        X_one_hot_list.append(X_f_one_hot)
    # (F, N, 2)
    X_one_hot = torch.stack(X_one_hot_list, dim=0).float()

    E = torch.zeros(N, N)
    E[dst, src] = 1.
    # (N, N, 2)
    E_one_hot = F.one_hot(E.long()).float()

    # (F, 2)
    X_one_hot_count = X_one_hot.sum(dim=1)
    # (F, 2)
    X_marginal = X_one_hot_count / X_one_hot_count.sum(dim=1, keepdim=True)

    # (N, C)
    Y_one_hot = F.one_hot(Y).float()
    # (C)
    Y_one_hot_count = Y_one_hot.sum(dim=0)
    # (C)
    Y_marginal = Y_one_hot_count / Y_one_hot_count.sum()

    # (2)
    E_one_hot_count = E_one_hot.sum(dim=0).sum(dim=0)
    E_marginal = E_one_hot_count / E_one_hot_count.sum()

    # P(X_f | Y)
    X_cond_Y_marginals = []
    num_classes = Y_marginal.size(-1)
    for k in range(num_classes):
        nodes_k = Y == k
        X_one_hot_k = X_one_hot[:, nodes_k]
        # (F, 2)
        X_one_hot_k_count = X_one_hot_k.sum(dim=1)
        # (F, 2)
        X_marginal_k = X_one_hot_k_count / X_one_hot_k_count.sum(dim=1, keepdim=True)
        X_cond_Y_marginals.append(X_marginal_k)
    # (F, C, 2)
    X_cond_Y_marginals = torch.stack(X_cond_Y_marginals, dim=1)

    return X_one_hot, Y, E_one_hot, X_marginal, Y_marginal, E_marginal, X_cond_Y_marginals
