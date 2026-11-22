import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from .data import IMAGENET_MEAN, IMAGENET_STD


def imagenet_val_loader(root, batch_size=256, num_workers=4, image_size=224):
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    dataset = datasets.ImageNet(root=root, split="val", transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)


@torch.no_grad()
def evaluate_primary(net, root, batch_size=256, num_workers=4, device="cuda"):
    loader = imagenet_val_loader(root, batch_size, num_workers)
    net = net.to(device).eval()
    dtype = next(net.parameters()).dtype
    correct = 0
    total = 0
    for images, labels in tqdm(loader, desc="imagenet val"):
        images = images.to(device=device, dtype=dtype)
        labels = labels.to(device)
        predictions = net(images).argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.numel()
    accuracy = 100.0 * correct / total
    print(f"[eval] imagenet top-1 {accuracy:.2f}%")
    return accuracy


def similarity_gap_ratio(original, immunized):
    return (original - immunized) / original * 100.0
