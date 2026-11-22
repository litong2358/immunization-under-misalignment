from types import SimpleNamespace

import torch
from torch.func import functional_call

from .losses import LabelSmoothingCrossEntropy

EPS = 1e-12


def flatten(params):
    return torch.cat([tensor.reshape(-1) for tensor in params.values()])


def norm(vector):
    return torch.linalg.vector_norm(vector) + EPS


def unit(vector):
    return vector / norm(vector)


def clone_params(module, prefix):
    return {
        f"{prefix}.{name}": param.detach().clone().requires_grad_(True)
        for name, param in module.named_parameters()
    }


class FunctionalModel:
    def __init__(self, net):
        self.net = net
        live = {
            f"feature_extractor.{name}": param
            for name, param in net.feature_extractor.named_parameters()
            if param.requires_grad
        }
        self.params = {
            **live,
            **clone_params(net.classifier, "classifier"),
            **clone_params(net.harmful_head, "harmful_head"),
        }
        self.criterion = LabelSmoothingCrossEntropy()

    def _split(self, params):
        groups = {"feature_extractor": {}, "classifier": {}, "harmful_head": {}}
        for key, value in params.items():
            prefix, name = key.split(".", 1)
            groups[prefix][name] = value
        return groups

    def logits(self, params, x, head):
        groups = self._split(params)
        features = functional_call(self.net.feature_extractor, groups["feature_extractor"], (x,))
        key = _head_key(head)
        module = self.net.harmful_head if key == "harmful_head" else self.net.classifier
        return functional_call(module, groups[key], (features,))

    def loss(self, params, x, y, head):
        return self.criterion(self.logits(params, x, head), y)

    def grad(self, params, x, y, head):
        keys = list(params.keys())
        loss = self.loss(params, x, y, head)
        grads = torch.autograd.grad(
            loss, [params[key] for key in keys], create_graph=True, retain_graph=True, allow_unused=True
        )
        return {
            key: grad if grad is not None else torch.zeros_like(params[key])
            for key, grad in zip(keys, grads)
        }


def _head_key(head):
    return "harmful_head" if head == "harmful" else "classifier"


def _thin_svd(matrix):
    if matrix.is_cuda:
        last = None
        for driver in ("gesvda", "gesvd"):
            try:
                return torch.linalg.svd(matrix, full_matrices=False, driver=driver)
            except Exception as error:
                last = error
        raise last
    return torch.linalg.svd(matrix, full_matrices=False)


class TrajectoryProbes:
    def __init__(self, net, inner_lr, subspace_rank):
        self.fn = FunctionalModel(net)
        self.inner_lr = inner_lr
        self.subspace_rank = subspace_rank
        self.scales = {
            key: torch.sqrt(param.detach().pow(2).mean() + EPS)
            for key, param in self.fn.params.items()
        }

    @property
    def params(self):
        return self.fn.params

    def loss(self, params, x, y, head):
        return self.fn.loss(params, x, y, head)

    def update_map(self, params, x, y, head):
        grads = self.fn.grad(params, x, y, head)
        step = {key: -self.inner_lr * grad for key, grad in grads.items()}
        return step, grads

    def rollout(self, x, y, head, steps):
        params = self.fn.params
        anchors, updates, grads = [], [], []
        for _ in range(steps):
            step, grad = self.update_map(params, x, y, head)
            anchors.append(params)
            updates.append(step)
            grads.append(grad)
            params = {key: params[key] + step[key] for key in params}
        return SimpleNamespace(anchors=anchors, updates=updates, grads=grads, final=params)

    def _columns(self, dicts):
        columns = []
        for entry in dicts:
            columns.append(
                torch.cat(
                    [entry[key].reshape(-1) / (self.scales[key] + EPS) for key in self.fn.params]
                )
            )
        return torch.stack(columns, dim=1)

    def _basis(self, dicts):
        matrix = self._columns(dicts)
        left, singular, _ = _thin_svd(matrix)
        keep = min(self.subspace_rank, int((singular > 1e-8).sum().item()))
        if keep == 0:
            return matrix.new_zeros(matrix.shape[0], 0)
        return left[:, :keep]

    def projectors(self, harmful_updates, primary_updates):
        harmful_basis = self._basis(harmful_updates)
        primary_basis = self._basis(primary_updates) if primary_updates is not None else None
        if primary_basis is None:
            return _projector(harmful_basis), None
        return (
            _projector(_orthogonal_complement(harmful_basis, primary_basis)),
            _projector(_orthogonal_complement(primary_basis, harmful_basis)),
        )

    def scale_down(self, vector):
        chunks, index = [], 0
        for key, param in self.fn.params.items():
            size = param.numel()
            chunks.append(vector[index:index + size] / (self.scales[key] + EPS))
            index += size
        return torch.cat(chunks)

    def probe_direction(self, update, projector):
        return unit(self.scale_down(projector(flatten(update))))

    def unit_direction(self, vector, projector):
        projected = projector(vector)
        if torch.count_nonzero(projected).item() == 0:
            return unit(vector)
        return unit(projected)

    def random_direction(self, projector):
        vector = torch.cat([torch.randn_like(param).reshape(-1) for param in self.fn.params.values()])
        return self.unit_direction(vector, projector)

    def _offset(self, direction, scale):
        offset, index, squared = {}, 0, None
        for key, param in self.fn.params.items():
            size = param.numel()
            block = direction[index:index + size].reshape_as(param) * (scale * (self.scales[key] + EPS))
            index += size
            offset[key] = block
            squared = block.pow(2).sum() if squared is None else squared + block.pow(2).sum()
        return offset, torch.sqrt(squared + EPS)

    def curvature(self, anchor, direction, x, y, head, delta):
        offset, offset_norm = self._offset(direction, delta)
        plus = {key: anchor[key] + offset[key] for key in anchor}
        minus = {key: anchor[key] - offset[key] for key in anchor}
        center = self.fn.loss(anchor, x, y, head)
        return (
            self.fn.loss(plus, x, y, head) - 2.0 * center + self.fn.loss(minus, x, y, head)
        ) / offset_norm.pow(2)

    def lipschitz(self, anchor, direction, x, y, head, eps):
        offset, offset_norm = self._offset(direction, eps)
        plus = {key: anchor[key] + offset[key] for key in anchor}
        shifted = self.fn.grad(plus, x, y, head)
        base = self.fn.grad(anchor, x, y, head)
        difference = torch.cat([(shifted[key] - base[key]).reshape(-1) for key in base])
        return norm(difference) / offset_norm

    def contractivity(self, anchor, anchor_step, directions, x, y, head, eps, power_iters, projector):
        base = {key: anchor[key] + anchor_step[key] for key in anchor}
        best = None
        index = None
        for position, direction in enumerate(directions):
            probe = direction
            for _ in range(power_iters):
                secant, _ = self._secant(anchor, base, probe, x, y, head, eps)
                probe = self.unit_direction(secant, projector)
            _, value = self._secant(anchor, base, probe, x, y, head, eps)
            if best is None or value > best:
                best = value
                index = position
        return best, index

    def _secant(self, anchor, base, direction, x, y, head, eps):
        offset, offset_norm = self._offset(direction, eps)
        shifted = {key: anchor[key] + offset[key] for key in anchor}
        step, _ = self.update_map(shifted, x, y, head)
        difference = torch.cat(
            [(shifted[key] + step[key] - base[key]).reshape(-1) for key in anchor]
        )
        return difference, norm(difference) / offset_norm


def _projector(basis):
    if basis is None or basis.shape[1] == 0:
        return lambda vector: vector
    return lambda vector: basis @ (basis.T @ vector)


def _orthogonal_complement(basis, other):
    if basis.shape[1] == 0 or other.shape[1] == 0:
        return basis
    residual = basis - other @ (other.T @ basis)
    if residual.shape[1] == 0:
        return basis
    return torch.linalg.qr(residual, mode="reduced").Q
