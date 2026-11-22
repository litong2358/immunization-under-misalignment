import argparse
import os

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

from clamp.config import experiment_dir, load_config, mirror_stdout, save_config, set_seed
from clamp.data import ImmunizationData
from clamp.module import CLAMP

torch.set_float32_matmul_precision("medium")


class EpochCheckpoint(pl.Callback):
    def __init__(self, directory):
        super().__init__()
        os.makedirs(directory, exist_ok=True)
        self.directory = directory

    def on_train_epoch_end(self, trainer, module):
        path = os.path.join(self.directory, f"epoch{trainer.current_epoch}.ckpt")
        trainer.save_checkpoint(path)
        print(f"[immunize] saved {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--harmful-head", default=None)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--results", default="results")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(args.seed)

    output_dir = experiment_dir(config, args.seed, args.results)
    save_config(config, os.path.join(output_dir, "config.yaml"))
    mirror_stdout(os.path.join(output_dir, "log.txt"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CLAMP(config).to(device=device, dtype=torch.double)

    head_path = args.harmful_head or config["model"].get("harmful_head")
    if head_path:
        model.load_harmful_head(head_path)
    print(model.net.describe())

    data = ImmunizationData(config["data"], debug=args.debug)

    checkpoint = ModelCheckpoint(
        dirpath=os.path.join(output_dir, "checkpoints"),
        filename="epoch={epoch}",
        auto_insert_metric_name=False,
        save_top_k=-1,
        every_n_epochs=1,
        save_on_train_epoch_end=True,
        save_last=True,
    )

    trainer = pl.Trainer(
        max_epochs=int(config["training"]["epochs"]),
        devices=1,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        precision=32,
        logger=CSVLogger(output_dir, name="logs"),
        callbacks=[checkpoint, EpochCheckpoint(os.path.join(output_dir, "checkpoints"))],
        log_every_n_steps=1,
        enable_checkpointing=True,
        num_sanity_val_steps=0,
    )
    trainer.fit(model, datamodule=data)
    print(f"[immunize] checkpoints in {os.path.join(output_dir, 'checkpoints')}")


if __name__ == "__main__":
    main()
