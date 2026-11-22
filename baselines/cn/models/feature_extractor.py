import torch.nn as nn
import torch
from models.resnet import resnet18
from models.linear import Linear
from timm.models import create_model

from pdb import set_trace as stx


class PretrainedFeatureExtractor(nn.Module):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset
        if dataset == "cifar10":
            base_model = resnet18(pretrained=True, num_classes=10)
            self.frozen_extractor = nn.Sequential(*list(base_model.children())[:-3])
            self.feature_extractor = nn.Sequential(
                *list(base_model.children())[-3:-1],
                nn.Flatten(),
            )
        elif dataset == "imagenet":
            base_model = create_model("resnet18", pretrained=True)
            self.frozen_extractor = nn.Sequential(*list(base_model.children())[:-3])
            for param in self.frozen_extractor.parameters():
                param.requires_grad = False
            self.feature_extractor = nn.Sequential(*list(base_model.children())[-3:-1])
            self.classifier = nn.Sequential(*list(base_model.children())[-1:])
        else:
            raise ValueError(f"Dataset {dataset} not supported.")
    def forward(self, x):
        x = self.frozen_extractor(x)
        x = self.feature_extractor(x)
        return x 
    


class LinearFeatureExtractor(nn.Module):
    def __init__(self, input_size, output_size, ckpt_path=None, preconditioner=False):
        super().__init__()
        self.preconditioner = preconditioner
        if preconditioner:
            self.feature_extractor = Linear(input_size, output_size)
        else:
            self.feature_extractor = nn.Linear(input_size, output_size)

        if ckpt_path is not None:
            checkpoint = torch.load(ckpt_path)
            new_state_dict = {}
            for key, value in checkpoint.items():
                if "feature_extractor.feature_extractor" in key:
                    new_key = key.replace("feature_extractor.feature_extractor", "feature_extractor")
                    new_state_dict[new_key] = value
            self.load_state_dict(new_state_dict)
    
    def forward(self, x, lambda_reg=1.0, use_precond=False):
        x = x.view(x.size(0), -1)
        if self.preconditioner:
            return self.feature_extractor(x, lambda_reg, use_precond)
        else:
            return self.feature_extractor(x)
    
class MLPFeatureExtractor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.feature_extractor(x)
