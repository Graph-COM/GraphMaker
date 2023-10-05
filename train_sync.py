import pandas as pd
import torch
import wandb

from data import load_dataset
from setup_utils import load_train_yaml

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

if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, required=True,
                        choices=["cora", "citeseer", "amazon_photo", "amazon_computer"])
    args = parser.parse_args()

    main(args)
