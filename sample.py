import dgl
import torch
import torch.nn.functional as F

from data import load_dataset, preprocess
from eval_utils import Evaluator
from setup_utils import set_seed

def main(args):
    state_dict = torch.load(args.model_path)
    dataset = state_dict["dataset"]

    train_yaml_data = state_dict["train_yaml_data"]
    model_name = train_yaml_data["meta_data"]["variant"]

    print(f"Loaded GraphMaker-{model_name} model trained on {dataset}")
    print(f"Val Nll {state_dict['best_val_nll']}")

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    g_real = load_dataset(dataset)
    X_one_hot_3d_real, Y_real, E_one_hot_real,\
        X_marginal, Y_marginal, E_marginal, X_cond_Y_marginals = preprocess(g_real)
    Y_one_hot_real = F.one_hot(Y_real)

    evaluator = Evaluator(dataset,
                          g_real,
                          X_one_hot_3d_real,
                          Y_one_hot_real)

    X_marginal = X_marginal.to(device)
    Y_marginal = Y_marginal.to(device)
    E_marginal = E_marginal.to(device)
    X_cond_Y_marginals = X_cond_Y_marginals.to(device)
    num_nodes = Y_real.size(0)

    if model_name == "Sync":
        from model import ModelSync

        model = ModelSync(X_marginal=X_marginal,
                          Y_marginal=Y_marginal,
                          E_marginal=E_marginal,
                          gnn_X_config=train_yaml_data["gnn_X"],
                          gnn_E_config=train_yaml_data["gnn_E"],
                          num_nodes=num_nodes,
                          **train_yaml_data["diffusion"]).to(device)

        model.graph_encoder.pred_X.load_state_dict(state_dict["pred_X_state_dict"])
        model.graph_encoder.pred_E.load_state_dict(state_dict["pred_E_state_dict"])

    elif model_name == "Async":
        from model import ModelAsync

        model = ModelAsync(X_marginal=X_marginal,
                           Y_marginal=Y_marginal,
                           E_marginal=E_marginal,
                           mlp_X_config=train_yaml_data["mlp_X"],
                           gnn_E_config=train_yaml_data["gnn_E"],
                           num_nodes=num_nodes,
                           **train_yaml_data["diffusion"]).to(device)

        model.graph_encoder.pred_X.load_state_dict(state_dict["pred_X_state_dict"])
        model.graph_encoder.pred_E.load_state_dict(state_dict["pred_E_state_dict"])

    model.eval()

    # Set seed for better reproducibility.
    set_seed()

    for _ in range(args.num_samples):
        X_0_one_hot, Y_0_one_hot, E_0 = model.sample()
        src, dst = E_0.nonzero().T
        g_sample = dgl.graph((src, dst), num_nodes=num_nodes).cpu()

        evaluator.add_sample(g_sample,
                             X_0_one_hot.cpu(),
                             Y_0_one_hot.cpu())

    evaluator.summary()

if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--model_path", type=str, help="Path to the model.")
    parser.add_argument("--num_samples", type=int, default=10,
                        help="Number of samples to generate.")
    args = parser.parse_args()

    main(args)
