import argparse
import json
import os

import torch

from clamp.adaptation import adapt
from clamp.config import load_config, set_seed
from clamp.data import ImmunizationData
from clamp.evaluation import evaluate_primary
from clamp.model import ImmunizedResNet


def adaptation_data(config):
    data = dict(config["data"])
    data["harmful_train_split"] = data.get("adaptation_train_split", data["harmful_train_split"])
    return ImmunizationData(data)


def build_model(config, checkpoint, device, dtype):
    net = ImmunizedResNet(
        num_trainable_blocks=config["model"]["num_trainable_blocks"],
        harmful_num_classes=config["data"]["harmful_num_classes"],
        arch=config["model"].get("arch", "resnet18"),
    )
    if checkpoint:
        net.load_immunized_state(checkpoint)
    return net.to(device=device, dtype=dtype)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--benign-config", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--skip-primary", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = getattr(torch, config["adaptation"].get("dtype", "float32"))
    os.makedirs(args.output, exist_ok=True)

    summary = {"checkpoint": args.checkpoint}

    if not args.skip_primary:
        net = build_model(config, args.checkpoint, device, dtype)
        summary["primary_accuracy"] = evaluate_primary(
            net, config["data"]["primary_root"], num_workers=config["data"].get("num_workers", 4)
        )
        del net
        torch.cuda.empty_cache()

    harmful_data = adaptation_data(config)
    train_loader, val_loader = harmful_data.harmful_dataloaders()
    net = build_model(config, args.checkpoint, device, dtype)
    harmful = adapt(
        net,
        train_loader,
        val_loader,
        config["data"]["harmful_num_classes"],
        config["adaptation"],
        args.output,
        device=device,
        tag="harmful",
    )
    summary["harmful_final_accuracy"] = float(harmful["val_accuracy"].iloc[-1])
    del net
    torch.cuda.empty_cache()

    if args.benign_config:
        benign_config = load_config(args.benign_config)
        benign_data = adaptation_data(benign_config)
        train_loader, val_loader = benign_data.harmful_dataloaders()
        net = build_model(config, args.checkpoint, device, dtype)
        benign = adapt(
            net,
            train_loader,
            val_loader,
            benign_config["data"]["harmful_num_classes"],
            config["adaptation"],
            args.output,
            device=device,
            tag="benign",
        )
        summary["benign_final_accuracy"] = float(benign["val_accuracy"].iloc[-1])

    with open(os.path.join(args.output, "summary.json"), "w") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
