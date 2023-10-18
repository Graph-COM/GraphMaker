import numpy as np
import os
import torch
import torch.nn as nn

class CN(nn.Module):
    def __init__(self, batch_size = 65536):
        super().__init__()

        self.best_threshold = nn.Parameter(torch.tensor(0), requires_grad=False)
        self.batch_size = batch_size

    def fit(self, A_train, A_full, val_mask):
        A_train = A_train.to_dense()
        A_full = A_full.to_dense()

        val_src, val_dst = val_mask.nonzero().T
        label = A_full[val_src, val_dst]
        label[label != 0] = 1
        label = label.cpu()

        A_train[A_train != 0.] = 1.

        num_batches = len(val_src) // self.batch_size
        if len(val_src) % self.batch_size != 0:
            num_batches += 1

        start = 0
        pred_list = []
        for i in range(num_batches):
            end = start + self.batch_size

            batch_src = val_src[start:end]
            batch_dst = val_dst[start:end]
            batch_pred = (A_train[batch_src] * A_train[batch_dst]).sum(dim=-1)
            batch_pred = batch_pred.cpu()
            pred_list.append(batch_pred)

            start = end

        pred = torch.cat(pred_list)

        thresholds = pred.unique()
        acc_list = []
        for bar in thresholds:
            pred_bar = (pred >= bar).float()
            acc_bar = (pred_bar == label).float().mean()
            acc_list.append(acc_bar.item())

        self.best_threshold = nn.Parameter(
            thresholds[np.argmax(acc_list)], requires_grad=False)

    def predict(self, A_train, A_full, mask):
        A_train = A_train.to_dense()
        A_full = A_full.to_dense()

        src, dst = mask.nonzero().T
        label = A_full[src, dst]
        label[label != 0] = 1
        label = label.cpu()

        A_train[A_train != 0.] = 1.

        num_batches = len(src) // self.batch_size
        if len(src) % self.batch_size != 0:
            num_batches += 1

        start = 0
        pred_list = []
        for i in range(num_batches):
            end = start + self.batch_size

            batch_src = src[start:end]
            batch_dst = dst[start:end]
            batch_pred = (A_train[batch_src] * A_train[batch_dst]).sum(dim=-1)

            batch_pred = batch_pred.cpu()
            batch_pred = (batch_pred >= self.best_threshold).float()
            pred_list.append(batch_pred)

            start = end

        pred = torch.cat(pred_list)

        return (pred == label).float().mean().item()

class CNEvaluator:
    def __init__(self,
                 model_path,
                 A_train,
                 A_full,
                 val_mask,
                 test_mask):
        self.real_A_train = A_train
        self.real_A_full = A_full
        self.real_test_mask = test_mask

        self.sample_sample_acc = []

        self.model_real = CN()
        if os.path.exists(model_path):
            self.model_real.load_state_dict(torch.load(model_path))
        else:
            self.model_real.fit(A_train, A_full, val_mask)
            torch.save(self.model_real.state_dict(), model_path)

        self.real_real_acc = self.model_real.predict(A_train, A_full, test_mask)

        self.real_sample_acc = []
        self.sample_real_acc = []
        self.sample_sample_acc = []

    def add_sample(self,
                   A_train,
                   A_full,
                   val_mask,
                   test_mask):
        self.real_sample_acc.append(
            self.model_real.predict(A_train, A_full, test_mask)
        )

        model_sample = CN()
        model_sample.fit(A_train, A_full, val_mask)

        self.sample_real_acc.append(
            model_sample.predict(
                self.real_A_train,
                self.real_A_full,
                self.real_test_mask)
        )

        self.sample_sample_acc.append(
            model_sample.predict(A_train, A_full, test_mask)
        )

    def summary(self):
        print(f"ACC(G|G): {self.real_real_acc}")
        mean_sample_real_acc = np.mean(self.sample_real_acc)
        print(f"ACC(G|G_hat): {mean_sample_real_acc}")

        mean_sample_sample_acc = np.mean(self.sample_sample_acc)
        print(f"ACC(G_hat|G_hat): {mean_sample_sample_acc}")
        mean_real_sample_acc = np.mean(self.real_sample_acc)
        print(f"ACC(G_hat|G): {mean_real_sample_acc}")
