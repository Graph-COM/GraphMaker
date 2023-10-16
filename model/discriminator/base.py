import itertools
import torch

class BaseTrainer:
    def __init__(self,
                 hyper_space,
                 search_priority_increasing,
                 patience):
        """Base class for training a discriminative model."""
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.hyper_space = hyper_space
        self.search_priority_increasing = search_priority_increasing
        self.patience = patience

    def get_config_list(self):
        vals = [self.hyper_space[k] for k in self.search_priority_increasing]

        config_list = []
        for items in itertools.product(*vals):
            items_dict = dict(zip(self.search_priority_increasing, items))
            config_list.append(items_dict)

        import ipdb
        ipdb.set_trace()

        return config_list

    def save_model(self, model_path):
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "model_config": self.best_model_config
        }, model_path)
