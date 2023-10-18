import torch
import torch.nn as nn

from copy import deepcopy
from tqdm import tqdm

from .gcn import GCNTrainer

class SGC(nn.Module):
    def __init__(self,
                 in_size,
                 out_size,
                 num_layers):
        super().__init__()

        self.lin = nn.Linear(in_size, out_size)
        self.num_layers = num_layers

    def forward(self, A, H):
        for _ in range(self.num_layers):
            H = A @ H
        return self.lin(H)

class SGCTrainer(GCNTrainer):
    def __init__(self, num_gnn_layers):
        hyper_space = {
            "lr": [3e-2, 1e-2, 3e-3],
            "num_layers": [num_gnn_layers]
        }
        search_priority_increasing = ["lr", "num_layers"]

        super().__init__(hyper_space=hyper_space,
                         search_priority_increasing=search_priority_increasing,
                         patience=5)

        self.num_gnn_layers = num_gnn_layers

    def fit_trial(self,
                  A,
                  X,
                  Y,
                  num_classes,
                  train_mask,
                  val_mask,
                  num_layers,
                  lr):
        model = SGC(in_size=X.size(1),
                    out_size=num_classes,
                    num_layers=num_layers).to(self.device)
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
                f"Training SGC {self.num_gnn_layers}-layer discriminator")

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
                        "num_layers": config['num_layers'],
                    }

                tconfig.set_postfix(accuracy=100. * best_acc)

                if trial_acc == 1.0:
                    break
        self.model = best_model
        self.best_model_config = best_model_config

    def load_model(self, model_path):
        state_dict = torch.load(model_path)
        model = SGC(**state_dict["model_config"]).to(self.device)
        model.load_state_dict(state_dict["model_state_dict"])
        self.model = model
