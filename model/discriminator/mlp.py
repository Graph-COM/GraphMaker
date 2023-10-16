import torch
import torch.nn as nn
import torch.nn.functional as F

from copy import deepcopy
from tqdm import tqdm

from .base import BaseTrainer

class MLP(nn.Module):
    def __init__(self,
                 in_size,
                 out_size,
                 num_layers,
                 hidden_size,
                 dropout):
        super().__init__()

        assert num_layers >= 2
        self.lins = nn.ModuleList()
        self.lins.append(nn.Linear(in_size, hidden_size))
        for _ in range(num_layers - 2):
            self.lins.append(nn.Linear(hidden_size, hidden_size))
        self.lins.append(nn.Linear(hidden_size, out_size))

        self.dropout = dropout

    def forward(self, h):
        for lin in self.lins[:-1]:
            h = lin(h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        return self.lins[-1](h)

class MLPTrainer(BaseTrainer):
    def __init__(self):
        hyper_space = {
            "lr": [3e-2, 1e-2, 3e-3],
            "num_layers": [2, 3],
            "hidden_size": [32, 128, 512],
            "dropout": [0., 0.1, 0.2]
        }
        search_priority_increasing = ["dropout", "lr", "num_layers", "hidden_size"]

        super().__init__(hyper_space=hyper_space,
                         search_priority_increasing=search_priority_increasing,
                         patience=5)

    def preprocess(self, X, Y):
        X = X.to(self.device).float()
        Y = Y.to(self.device)

        # row normalize
        X = F.normalize(X, p=1, dim=1)

        return X, Y

    def fit_trial(self,
                  X,
                  Y,
                  num_classes,
                  train_mask,
                  val_mask,
                  num_layers,
                  hidden_size,
                  dropout,
                  lr):
        model = MLP(in_size=X.size(1),
                    out_size=num_classes,
                    num_layers=num_layers,
                    hidden_size=hidden_size,
                    dropout=dropout).to(self.device)
        loss_func = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        num_epochs = 1000
        num_patient_epochs = 0
        best_acc = 0
        best_model_state_dict = model.state_dict()
        for epoch in range(1, num_epochs + 1):
            model.train()
            logits = model(X)
            loss = loss_func(logits[train_mask], Y[train_mask])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            acc = self.predict(X, Y, val_mask, model)

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

    def fit(self, X, Y, num_classes, train_mask, val_mask):
        """
        Parameters
        ----------
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
        X, Y = self.preprocess(X, Y)

        config_list = self.get_config_list()

        best_acc = 0
        with tqdm(config_list) as tconfig:
            tconfig.set_description(f"Training MLP discriminator")

            for config in tconfig:
                trial_acc, trial_model = self.fit_trial(X,
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
                        "num_layers": config["num_layers"],
                        "hidden_size": config["hidden_size"],
                        "dropout": config["dropout"]
                    }

                tconfig.set_postfix(accuracy=100. * best_acc)

                if trial_acc == 1.0:
                    break
        self.model = best_model
        self.best_model_config = best_model_config

    @torch.no_grad()
    def predict(self, X, Y, mask=None, model=None):
        X, Y = self.preprocess(X, Y)

        if model is None:
            model = self.model

        model.eval()

        if mask is None:
            logits = model(X)
            pred = logits.argmax(dim=-1, keepdim=True).reshape(-1)
            return (pred == Y).float().mean().item()
        else:
            logits = model(X[mask])
            pred = logits.argmax(dim=-1, keepdim=True).reshape(-1)
            return (pred == Y[mask]).float().mean().item()

    def load_model(self, model_path):
        state_dict = torch.load(model_path)
        model = MLP(**state_dict["model_config"]).to(self.device)
        model.load_state_dict(state_dict["model_state_dict"])
        self.model = model
