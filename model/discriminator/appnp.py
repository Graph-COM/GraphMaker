import dgl.sparse as dglsp
import torch
import torch.nn as nn
import torch.nn.functional as F

from copy import deepcopy
from tqdm import tqdm

from .gcn import GCNTrainer

class APPNP(nn.Module):
    def __init__(self,
                 in_size,
                 out_size,
                 num_trans_layers,
                 hidden_size,
                 dropout,
                 num_prop_layers,
                 alpha):
        super().__init__()

        assert num_trans_layers >= 2
        self.lins = nn.ModuleList()
        self.lins.append(nn.Linear(in_size, hidden_size))
        for _ in range(num_trans_layers - 2):
            self.lins.append(nn.Linear(hidden_size, hidden_size))
        self.lins.append(nn.Linear(hidden_size, out_size))

        self.dropout = dropout

        self.num_prop_layers = num_prop_layers
        self.alpha = alpha

    def forward(self, A, H):
        # Predict.
        for lin in self.lins[:-1]:
            H = lin(H)
            H = F.relu(H)
            H = F.dropout(H, p=self.dropout, training=self.training)
        H_local = self.lins[-1](H)

        # Propagate.
        H = H_local
        for _ in range(self.num_prop_layers):
            A_drop = dglsp.val_like(
                A, F.dropout(A.val, p=self.dropout, training=self.training))
            H = A_drop @ H + self.alpha * H_local
        return H

class APPNPTrainer(GCNTrainer):
    def __init__(self, num_gnn_layers):
        hyper_space = {
            "lr": [3e-2, 1e-2, 3e-3],
            "num_trans_layers": [2],
            "hidden_size": [32, 128, 512],
            "dropout": [0., 0.1],
            "num_prop_layers": [num_gnn_layers],
            "alpha": [0.1, 0.2]
        }
        search_priority_increasing = [
            "dropout",
            "alpha",
            "lr",
            "num_trans_layers",
            "num_prop_layers",
            "hidden_size"]

        super().__init__(num_gnn_layers=num_gnn_layers,
                         hyper_space=hyper_space,
                         search_priority_increasing=search_priority_increasing,
                         patience=5)

    def fit_trial(self,
                  A,
                  X,
                  Y,
                  num_classes,
                  train_mask,
                  val_mask,
                  num_trans_layers,
                  hidden_size,
                  dropout,
                  num_prop_layers,
                  alpha,
                  lr):
        model = APPNP(in_size=X.size(1),
                      out_size=num_classes,
                      num_trans_layers=num_trans_layers,
                      hidden_size=hidden_size,
                      dropout=dropout,
                      num_prop_layers=num_prop_layers,
                      alpha=alpha).to(self.device)
        loss_func = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        num_epochs = 1000
        num_patient_epochs = 0
        best_acc = 0
        best_model_state_dict = deepcopy(model.state_dict())
        for epoch in range(1, num_epochs + 1):
            model.train()
            logits = model(A, X)
            loss = loss_func(logits[train_mask], Y[train_mask])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            acc = self.predict(A, X, Y, val_mask, model)

            if acc > best_acc:
                num_patient_epochs = 0
                best_acc = acc
                best_model_state_dict = deepcopy(model.state_dict())
            else:
                num_patient_epochs += 1

            if num_patient_epochs == self.patience:
                break

        model.load_state_dict(best_model_state_dict)
        return best_acc, model

    def fit(self, A, X, Y, num_classes, train_mask, val_mask):
        """
        Parameters
        ----------
        A : dgl.sparse.SparseMatrix
            Adjacency matrix.
        X : torch.Tensor of shape (|V|, D)
            Binary node features.
        Y : torch.Tensor of shape (|V|,)
            Node labels.
        num_classes : int
            Number of node classes.
        train_mask : torch.Tensor of shape (|V|)
            Mask indicating training nodes.
        val_mask : torch.Tensor of shape (|V|)
            Mask indicating validation nodes.
        """
        A, X, Y = self.preprocess(A, X, Y)

        config_list = self.get_config_list()

        best_acc = 0
        with tqdm(config_list) as tconfig:
            tconfig.set_description(
                f"Training APPNP {self.num_gnn_layers}-layer discriminator")

            for config in tconfig:
                trial_acc, trial_model = self.fit_trial(A,
                                                        X,
                                                        Y,
                                                        num_classes,
                                                        train_mask,
                                                        val_mask,
                                                        **config)

                if trial_acc > best_acc:
                    best_acc = trial_acc
                    best_model = trial_model
                    best_model_config = {
                        "in_size": X.size(1),
                        "out_size": num_classes,
                        "num_trans_layers": config["num_trans_layers"],
                        "hidden_size": config["hidden_size"],
                        "dropout": config["dropout"],
                        "num_prop_layers": config["num_prop_layers"],
                        "alpha": config["alpha"]
                    }

                tconfig.set_postfix(accuracy=100. * best_acc)

                if trial_acc == 1.0:
                    break
        self.model = best_model
        self.best_model_config = best_model_config

    def load_model(self, model_path):
        state_dict = torch.load(model_path)
        model = APPNP(**state_dict["model_config"]).to(self.device)
        model.load_state_dict(state_dict["model_state_dict"])
        self.model = model
