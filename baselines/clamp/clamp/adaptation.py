import os

import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm


def configure_trainable(net, num_trainable_blocks):
    for param in net.parameters():
        param.requires_grad = False
    for param in net.classifier.parameters():
        param.requires_grad = True
    blocks = list(net.feature_extractor.children())
    frozen = len(blocks) - max(0, num_trainable_blocks)
    for index, block in enumerate(blocks):
        if index >= frozen:
            for param in block.parameters():
                param.requires_grad = True


def _run_epoch(net, loader, device, dtype, criterion, optimizer=None, desc=""):
    net.train(optimizer is not None)
    correct = 0
    total = 0
    running = 0.0
    batches = 0
    context = torch.enable_grad() if optimizer is not None else torch.no_grad()
    with context:
        for images, labels in tqdm(loader, desc=desc, leave=False):
            images = images.to(device=device, dtype=dtype)
            labels = labels.to(device)
            logits = net(images)
            loss = criterion(logits, labels)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            running += loss.item()
            batches += 1
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += labels.numel()
    return {"loss": running / max(batches, 1), "accuracy": 100.0 * correct / total}


def adapt(net, train_loader, val_loader, num_classes, config, output_dir, device="cuda", tag="dh"):
    os.makedirs(output_dir, exist_ok=True)
    net = net.to(device)
    net.replace_classifier(num_classes)
    configure_trainable(net, int(config["num_trainable_blocks"]))
    dtype = next(net.parameters()).dtype

    optimizer = torch.optim.SGD(
        [p for p in net.parameters() if p.requires_grad],
        lr=float(config["lr"]),
        momentum=float(config["momentum"]),
        weight_decay=float(config["weight_decay"]),
    )
    criterion = nn.CrossEntropyLoss()

    epochs = int(config["epochs"])
    sched = config.get("schedule", "constant")
    warmup = int(config.get("warmup_epochs", 0))
    min_lr = float(config.get("min_lr", 0.0))
    base_lr = float(config["lr"])
    milestones = [int(m) for m in config.get("milestones", [])]
    gamma = float(config.get("gamma", 0.1))
    assert sched in ("constant", "cosine", "cos_decay", "multistep"), sched

    def lr_at(ep):
        if sched == "cos_decay":
            import math as _m
            return min_lr + 0.5 * (base_lr - min_lr) * (1 + _m.cos(_m.pi * (ep - 1) / epochs))
        if sched == "multistep":
            return base_lr * gamma ** sum(ep - 1 >= m for m in milestones)
        if sched != "cosine":
            return base_lr
        if ep <= warmup:
            return base_lr * ep / max(warmup, 1)
        import math as _m
        t = (ep - warmup) / max(epochs - warmup, 1)
        return min_lr + 0.5 * (base_lr - min_lr) * (1 + _m.cos(_m.pi * t))

    rows = []
    path = os.path.join(output_dir, f"{tag}_adaptation.csv")
    for epoch in range(epochs):
        for g in optimizer.param_groups:
            g["lr"] = lr_at(epoch + 1)
        train = _run_epoch(net, train_loader, device, dtype, criterion, optimizer, f"{tag} train {epoch + 1}")
        val = _run_epoch(net, val_loader, device, dtype, criterion, None, f"{tag} val {epoch + 1}")
        rows.append(
            {
                "epoch": epoch + 1,
                "train_loss": train["loss"],
                "train_accuracy": train["accuracy"],
                "val_loss": val["loss"],
                "val_accuracy": val["accuracy"],
            }
        )
        pd.DataFrame(rows).to_csv(path, index=False)
        print(
            f"[{tag}] epoch {epoch + 1}/{config['epochs']} "
            f"train {train['accuracy']:.2f}% val {val['accuracy']:.2f}%"
        )
    return pd.DataFrame(rows)
