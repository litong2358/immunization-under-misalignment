import copy
import random

import torch


class PCGrad:
    def __init__(self, optimizer):
        self.optimizer = optimizer

    def _parameters(self):
        return [param for group in self.optimizer.param_groups for param in group["params"]]

    def _pack(self, objectives):
        parameters = self._parameters()
        gradients, masks = [], []
        for objective in objectives:
            self.optimizer.zero_grad(set_to_none=True)
            objective.backward(retain_graph=True)
            flat, mask = [], []
            for param in parameters:
                if param.grad is None:
                    flat.append(torch.zeros_like(param).reshape(-1))
                    mask.append(torch.zeros_like(param).reshape(-1))
                else:
                    flat.append(param.grad.detach().clone().reshape(-1))
                    mask.append(torch.ones_like(param).reshape(-1))
            gradients.append(torch.cat(flat))
            masks.append(torch.cat(mask))
        return gradients, masks

    def _project(self, gradients, masks):
        shared = torch.stack(masks).prod(0).bool()
        projected = copy.deepcopy(gradients)
        for target in projected:
            order = list(gradients)
            random.shuffle(order)
            for other in order:
                overlap = torch.dot(target, other)
                if overlap < 0:
                    target -= overlap * other / (other.norm() ** 2 + 1e-12)
        merged = torch.zeros_like(gradients[0])
        merged[shared] = torch.stack([g[shared] for g in projected]).mean(dim=0)
        merged[~shared] = torch.stack([g[~shared] for g in projected]).sum(dim=0)
        return merged

    def backward(self, objectives):
        gradients, masks = self._pack(objectives)
        merged = self._project(gradients, masks)
        index = 0
        for param in self._parameters():
            size = param.numel()
            param.grad = merged[index:index + size].reshape_as(param).clone()
            index += size

    def zero_grad(self):
        self.optimizer.zero_grad(set_to_none=True)

    def step(self):
        self.optimizer.step()
