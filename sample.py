import torch
import torch.nn.functional as F

from data import load_dataset, preprocess
from eval_utils import Evaluator

def main(args):
    state_dict = torch.load(args.model_path)
    dataset = state_dict["dataset"]

    train_yaml_data = state_dict["train_yaml_data"]
    model_name = train_yaml_data["meta_data"]["variant"]

    print(f"Loaded GraphMaker-{model_name} model trained on {dataset}")
    print(f"Val Nll {state_dict['best_val_nll']}")

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    g_real = load_dataset(args.dataset)
    X_one_hot_3d_real, Y_real, E_one_hot_real,\
        X_marginal, Y_marginal, E_marginal, X_cond_Y_marginals = preprocess(g_real)
    Y_one_hot_real = F.one_hot(Y_real)

    evaluator = Evaluator(dataset,
                          g_real,
                          X_one_hot_3d_real,
                          Y_one_hot_real)

if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--model_path", type=str, help="Path to the model.")
    parser.add_argument("--num_samples", type=int, default=10,
                        help="Number of samples to generate.")
    args = parser.parse_args()

    main(args)
