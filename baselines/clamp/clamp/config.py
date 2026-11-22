import os
import random
import sys

import numpy as np
import pytorch_lightning as pl
import torch
import yaml


def load_config(path):
    with open(path) as handle:
        return yaml.safe_load(handle)


def save_config(config, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    pl.seed_everything(seed, workers=True)


def experiment_dir(config, seed, root="results"):
    data = config["data"]
    path = os.path.join(root, data["harmful_name"], config["name"], str(seed))
    os.makedirs(path, exist_ok=True)
    return path


class Tee:
    def __init__(self, stream, path):
        self.stream = stream
        self.file = open(path, "a")

    def write(self, message):
        self.stream.write(message)
        self.file.write(message)
        self.flush()

    def flush(self):
        self.stream.flush()
        self.file.flush()


def mirror_stdout(path):
    tee = Tee(sys.stdout, path)
    sys.stdout = tee
    sys.stderr = tee
    return tee
