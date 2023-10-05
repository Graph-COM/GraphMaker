import dgl

from dgl.data import AmazonCoBuyPhotoDataset, AmazonCoBuyComputerDataset, \
    CoraGraphDataset, CiteseerGraphDataset

def load_dataset(data_name):
    if data_name == "cora":
        dataset = CoraGraphDataset()
    elif data_name == "citeseer":
        dataset = CiteseerGraphDataset()
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
