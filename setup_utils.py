import yaml

def load_train_yaml(data_name, model_name):
    with open(f"configs/{data_name}/train_{model_name}.yaml") as f:
        yaml_data = yaml.load(f, Loader=yaml.loader.SafeLoader)
