import torch

class Evaluator:
    def __init__(self,
                 data_name,
                 dgl_g_real,
                 X_one_hot_3d_real,
                 Y_one_hot_real):
        """
        Parameters
        ----------
        data_name : str
            Name of the dataset.
        dgl_g_real : dgl.DGLGraph
            Real graph.
        """
        self.data_name = data_name

        # If the number of edges in a newly added graph exceeds this limit,
        # a subgraph will be used for certain metric computations.
        self.edge_limit = min(dgl_g_real.num_edges(), 20000)

        # Split datasets without a built-in split.
        add_mask = False
        if data_name in ["amazon_photo", "amazon_computer"]:
            add_mask = True
            torch.manual_seed(0)

        self.preprocess_g(dgl_g_real,
                          X_one_hot_3d_real,
                          Y_one_hot_real,
                          add_mask)

    def add_mask_cora(self, dgl_g, Y_one_hot):
        pass

    def add_mask_citeseer(self, dgl_g, Y_one_hot):
        pass

    def add_mask_benchmark(self, dgl_g, Y_one_hot):
        pass

    def add_mask(self, dgl_g, Y_one_hot):
        if self.data_name == "cora":
            return self.add_mask_cora(dgl_g, Y_one_hot)
        elif self.data_name == "citeseer":
            return self.add_mask_citeseer(dgl_g, Y_one_hot)
        elif self.data_name in ["amazon_photo", "amazon_computer"]:
            return self.add_mask_benchmark(dgl_g, Y_one_hot)
        else:
            raise ValueError(f'Unexpected data name: {self.data_name}')

    def preprocess_g(self,
                     dgl_g,
                     X_one_hot_3d,
                     Y_one_hot,
                     add_mask):
        """"""
        pass
