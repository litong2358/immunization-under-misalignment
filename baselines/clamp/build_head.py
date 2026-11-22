import argparse
import os

import torch
import torch.nn.functional as F
from tqdm import tqdm

from clamp.config import load_config, set_seed
from clamp.data import ImmunizationData
from clamp.model import ImmunizedResNet


@torch.no_grad()
def extract_features(net, loader, device, dtype):
    features, labels = [], []
    for images, targets in tqdm(loader, desc="features"):
        images = images.to(device=device, dtype=dtype)
        features.append(net.features(images).detach().cpu())
        labels.append(targets.detach().cpu())
    return torch.cat(features), torch.cat(labels)


def centroid_head(features, labels, num_classes, l2norm=True, temperature=1.0, device=None):
    target = device if device is not None else features.device
    features = features.to(target)
    labels = labels.to(target)
    if l2norm:
        features = F.normalize(features, p=2, dim=1)
    sums = torch.zeros(num_classes, features.shape[1], dtype=features.dtype, device=target)
    counts = torch.zeros(num_classes, dtype=torch.long, device=target)
    for label in range(num_classes):
        mask = labels == label
        if mask.any():
            sums[label] = features[mask].sum(dim=0)
            counts[label] = mask.sum()
    centroids = sums / counts.clamp_min(1).unsqueeze(1)
    if l2norm:
        centroids = F.normalize(centroids, p=2, dim=1)
    head = torch.nn.Linear(features.shape[1], num_classes, dtype=features.dtype)
    head.weight.data = (centroids / max(temperature, 1e-8)).cpu()
    head.bias.data.zero_()
    return head


@torch.no_grad()
def accuracy(head, features, labels):
    predictions = head(features).argmax(dim=1)
    return 100.0 * (predictions == labels).float().mean().item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=1.0)
    args = parser.parse_args()

    set_seed(args.seed)
    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net = ImmunizedResNet(
        num_trainable_blocks=config["model"]["num_trainable_blocks"],
        harmful_num_classes=config["data"]["harmful_num_classes"],
        arch=config["model"].get("arch", "resnet18"),
    ).to(device=device, dtype=torch.double).eval()

    data = ImmunizationData(config["data"])
    train_loader, val_loader = data.harmful_dataloaders(augment=False)

    train_features, train_labels = extract_features(net, train_loader, device, torch.double)
    head = centroid_head(
        train_features,
        train_labels,
        config["data"]["harmful_num_classes"],
        temperature=args.temperature,
        device=device,
    )
    print(f"[head] train accuracy {accuracy(head, train_features, train_labels):.2f}%")

    val_features, val_labels = extract_features(net, val_loader, device, torch.double)
    print(f"[head] val accuracy {accuracy(head, val_features, val_labels):.2f}%")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    torch.save({"state_dict": head.state_dict()}, args.output)
    print(f"[head] saved to {args.output}")


if __name__ == "__main__":
    main()
