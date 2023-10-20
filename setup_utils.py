import dgl
import numpy as np
import pydantic
import random
import torch
import yaml

from typing import Optional

# pydantic allows checking field types when loading configuration files
class MetaDataYaml(pydantic.BaseModel):
    variant: str

class GNNXYaml(pydantic.BaseModel):
    hidden_t: int
    hidden_X: int
    hidden_Y: int
    num_gnn_layers: int
    dropout: float

class GNNEYaml(pydantic.BaseModel):
    hidden_t: int
    hidden_X: int
    hidden_Y: int
    hidden_E: int
    num_gnn_layers: int
    dropout: float

class DiffusionYaml(pydantic.BaseModel):
    T: int

class OptimizerYaml(pydantic.BaseModel):
    lr: float
    weight_decay: Optional[float] = 0.
    amsgrad: Optional[bool] = False

class LRSchedulerYaml(pydantic.BaseModel):
    factor: float
    patience: int
    verbose: bool

class TrainYaml(pydantic.BaseModel):
    num_epochs: int
    val_every_epochs: int
    patient_epochs: int
    max_grad_norm: Optional[float] = None
    batch_size: int
    val_batch_size: int

class SyncYaml(pydantic.BaseModel):
    meta_data: MetaDataYaml
    gnn_X: GNNXYaml
    gnn_E: GNNEYaml
    diffusion: DiffusionYaml
    optimizer_X: OptimizerYaml
    optimizer_E: OptimizerYaml
    lr_scheduler: LRSchedulerYaml
    train: TrainYaml

class MLPXYaml(pydantic.BaseModel):
    hidden_t: int
    hidden_X: int
    hidden_Y: int
    num_mlp_layers: int
    dropout: float

class DiffusionAsyncYaml(pydantic.BaseModel):
    T_X: int
    T_E: int

class AsyncYaml(pydantic.BaseModel):
    meta_data: MetaDataYaml
    mlp_X: MLPXYaml
    gnn_E: GNNEYaml
    diffusion: DiffusionAsyncYaml
    optimizer_X: OptimizerYaml
    optimizer_E: OptimizerYaml
    lr_scheduler: LRSchedulerYaml
    train: TrainYaml

def load_train_yaml(data_name, model_name):
    with open(f"configs/{data_name}/train_{model_name}.yaml") as f:
        yaml_data = yaml.load(f, Loader=yaml.loader.SafeLoader)

    if model_name == "Sync":
        return SyncYaml(**yaml_data).model_dump()
    elif model_name == "Async":
        return AsyncYaml(**yaml_data).model_dump()

def set_seed(seed=0):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    dgl.seed(seed)
