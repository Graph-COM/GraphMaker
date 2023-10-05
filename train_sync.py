import pandas as pd
import torch
import wandb

from torch.utils.data import DataLoader

from data import load_dataset, preprocess
from model import BaseModel
from setup_utils import load_train_yaml, set_seed

def main(args):
    model_name = "Sync"
    yaml_data = load_train_yaml(args.dataset, model_name)

    config_df = pd.json_normalize(yaml_data, sep='/')
    # Number of time steps
    T = yaml_data['diffusion']['T']
    wandb.init(
        project=f"{args.dataset}-{model_name}",
        name=f"T{T}",
        config=config_df.to_dict(orient='records')[0])

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    g = load_dataset(args.dataset)
    X_one_hot_3d, Y, E_one_hot,\
        X_marginal, Y_marginal, E_marginal, X_cond_Y_marginals = preprocess(g)

    X_one_hot_3d = X_one_hot_3d.to(device)
    Y = Y.to(device)
    E_one_hot = E_one_hot.to(device)

    X_marginal = X_marginal.to(device)
    Y_marginal = Y_marginal.to(device)
    E_marginal = E_marginal.to(device)

    N = g.num_nodes()
    dst, src = torch.triu_indices(N, N, offset=1, device=device)
    # (|E|, 2), |E| for number of edges
    edge_index = torch.stack([dst, src], dim=1)

    # Set seed for better reproducibility.
    set_seed()

    train_config = yaml_data["train"]
    # For mini-batch training
    data_loader = DataLoader(edge_index.cpu(), batch_size=train_config["batch_size"],
                             shuffle=True, num_workers=4)
    val_data_loader = DataLoader(edge_index, batch_size=train_config["val_batch_size"],
                                 shuffle=False)

    # model =

if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, required=True,
                        choices=["cora", "citeseer", "amazon_photo", "amazon_computer"])
    args = parser.parse_args()

    main(args)
