import yaml
import torch

def load_config(config_path="config.yaml"):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    # Convert device string to torch.device
    config['device'] = torch.device(config['device'] if torch.cuda.is_available() else "cpu")
    return config

config = load_config()
