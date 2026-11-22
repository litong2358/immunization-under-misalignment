import torch
import torch.nn as nn
from timm.models import create_model


def _flatten_blocks(backbone):
    blocks = []
    for child in list(backbone.children())[:-1]:
        if isinstance(child, nn.Sequential):
            blocks.extend(list(child))
        else:
            blocks.append(child)
    return blocks


class ImmunizedResNet(nn.Module):
    def __init__(self, num_trainable_blocks=2, harmful_num_classes=196, arch="resnet18"):
        super().__init__()
        backbone = create_model(arch, pretrained=True)
        blocks = _flatten_blocks(backbone)
        self.feature_extractor = nn.Sequential(*blocks)
        self.classifier = list(backbone.children())[-1]
        self.feature_dim = self.classifier.in_features
        self.harmful_head = nn.Linear(self.feature_dim, harmful_num_classes)
        self.set_trainable_blocks(num_trainable_blocks)
        for param in self.harmful_head.parameters():
            param.requires_grad = False

    def set_trainable_blocks(self, num_trainable_blocks):
        blocks = list(self.feature_extractor.children())
        frozen = len(blocks) - max(0, num_trainable_blocks)
        for index, block in enumerate(blocks):
            for param in block.parameters():
                param.requires_grad = index >= frozen
        for param in self.classifier.parameters():
            param.requires_grad = True

    def describe(self):
        blocks = list(self.feature_extractor.children())
        lines = []
        for index, block in enumerate(blocks):
            count = sum(p.numel() for p in block.parameters())
            state = "trainable" if any(p.requires_grad for p in block.parameters()) else "frozen"
            lines.append(f"  [{index:2d}] {block.__class__.__name__:22s} {count:>10,} {state}")
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        lines.append(f"  trainable {trainable:,}/{total:,}")
        return "\n".join(lines)

    def forward(self, x):
        return self.classifier(self.feature_extractor(x))

    def features(self, x):
        return self.feature_extractor(x)

    def load_immunized_state(self, path):
        raw = torch.load(path, map_location="cpu", weights_only=False)
        state = raw.get("state_dict", raw)
        cleaned = {}
        for key, value in state.items():
            name = key
            for prefix in ("net.", "model.", "module."):
                if name.startswith(prefix):
                    name = name[len(prefix):]
            name = name.replace("feature_extractor.feature_extractor.", "feature_extractor.")
            cleaned[name] = value
        missing, unexpected = self.load_state_dict(cleaned, strict=False)
        print(f"[ckpt] loaded {path} missing={len(missing)} unexpected={len(unexpected)}")
        return self

    def replace_classifier(self, num_classes, weights=None):
        device = self.classifier.weight.device
        dtype = self.classifier.weight.dtype
        head = nn.Linear(self.feature_dim, num_classes).to(device=device, dtype=dtype)
        if weights is not None:
            raw = torch.load(weights, map_location="cpu", weights_only=False)
            head.load_state_dict(raw.get("state_dict", raw))
            head = head.to(device=device, dtype=dtype)
        self.classifier = head
        return head
