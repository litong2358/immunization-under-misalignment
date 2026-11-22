import yaml
import os
from datetime import datetime
import random
import numpy as np
import torch
import pytorch_lightning as pl

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    pl.seed_everything(seed)

def load_yaml(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

def merge_configs(*configs):
    merged = {}
    for config in configs:
        merged.update(config)
    return merged


def merge_args(base_config, cli_args):
    for key, value in vars(cli_args).items():
        if value is not None:
            keys = key.split('.')
            target = base_config
            for subkey in keys[:-1]:
                target = target.get(subkey, {})
            target[keys[-1]] = value
    return base_config


def save_config(config, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as file:
        yaml.dump(config, file)
    print(f"Configuration saved to {output_path}")


def create_experiment_dir(config, seed, base_dir="results"):
    dataset_name = config['data']['dataset_name']
    if dataset_name == 'imagenet':
        dataset_name += '_' + config['data']['d2_name']
    task = config['model'].get('task', 'classification')
    extractor_type = config['model']['feature_extractor_type']

    experiment_dir = os.path.join(
        base_dir,
        dataset_name,
        task,
        extractor_type,
        str(seed)
    )

    os.makedirs(experiment_dir, exist_ok=True)

    config_path = os.path.join(experiment_dir, "config.yaml")
    save_config(config, config_path)

    return experiment_dir


def get_config_value(config, key):
    value = config.get(key, None)
    return float(value) if value is not None else None
