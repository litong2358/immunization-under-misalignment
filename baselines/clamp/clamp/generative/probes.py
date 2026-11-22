import torch

EPS = 1e-12


def flatten(params):
    return torch.cat([tensor.reshape(-1) for tensor in params.values()])


def norm(vector):
    return torch.linalg.vector_norm(vector) + EPS


def unit(vector):
    return vector / norm(vector)


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


def _projector(basis):
    if basis is None or basis.shape[1] == 0:
        return lambda vector: vector
    return lambda vector: basis @ (basis.T @ vector)


class Rollout:
    def __init__(self, anchors, updates, grads, final):
        self.anchors, self.updates, self.grads, self.final = anchors, updates, grads, final


class DiffusionTrajectoryProbes:
    def __init__(self, phi, loss_fn, inner_lr, subspace_rank):
        self.phi = phi
        self.loss_fn = loss_fn
        self.inner_lr = inner_lr
        self.subspace_rank = subspace_rank
        self.scales = {
            key: torch.sqrt(param.detach().pow(2).mean() + EPS) for key, param in phi.items()
        }

    @property
    def params(self):
        return self.phi

    def loss(self, params):
        return self.loss_fn(params)

    def grad(self, params):
        keys = list(params.keys())
        loss = self.loss_fn(params)
        grads = torch.autograd.grad(
            loss, [params[key] for key in keys],
            create_graph=True, retain_graph=True, allow_unused=True,
        )
        return {k: (g if g is not None else torch.zeros_like(params[k]))
                for k, g in zip(keys, grads)}

    def update_map(self, params):
        grads = self.grad(params)
        step = {key: -self.inner_lr * grad for key, grad in grads.items()}
        return step, grads

    def rollout(self, steps):
        params = self.phi
        anchors, updates, grads = [], [], []
        for _ in range(steps):
            step, grad = self.update_map(params)
            anchors.append(params)
            updates.append(step)
            grads.append(grad)
            params = {key: params[key] + step[key] for key in params}
        return Rollout(anchors, updates, grads, params)

    def _columns(self, dicts):
        columns = [
            torch.cat([entry[key].reshape(-1) / (self.scales[key] + EPS) for key in self.phi])
            for entry in dicts
        ]
        return torch.stack(columns, dim=1)

    def _basis(self, dicts):
        matrix = self._columns(dicts)
        left, singular, _ = _thin_svd(matrix)
        keep = min(self.subspace_rank, int((singular > 1e-8).sum().item()))
        if keep == 0:
            return matrix.new_zeros(matrix.shape[0], 0)
        return left[:, :keep]

    def projector(self, updates):
        return _projector(self._basis(updates))

    def scale_down(self, vector):
        chunks, index = [], 0
        for key, param in self.phi.items():
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
        vector = torch.cat([torch.randn_like(p).reshape(-1) for p in self.phi.values()])
        return self.unit_direction(vector, projector)

    def _offset(self, direction, scale):
        offset, index, squared = {}, 0, None
        for key, param in self.phi.items():
            size = param.numel()
            block = direction[index:index + size].reshape_as(param) * (scale * (self.scales[key] + EPS))
            index += size
            offset[key] = block
            squared = block.pow(2).sum() if squared is None else squared + block.pow(2).sum()
        return offset, torch.sqrt(squared + EPS)

    def curvature(self, anchor, direction, delta):
        offset, offset_norm = self._offset(direction, delta)
        plus = {k: anchor[k] + offset[k] for k in anchor}
        minus = {k: anchor[k] - offset[k] for k in anchor}
        center = self.loss_fn(anchor)
        return (self.loss_fn(plus) - 2.0 * center + self.loss_fn(minus)) / offset_norm.pow(2)

    def lipschitz(self, anchor, direction, eps):
        offset, offset_norm = self._offset(direction, eps)
        plus = {k: anchor[k] + offset[k] for k in anchor}
        shifted = self.grad(plus)
        base = self.grad(anchor)
        difference = torch.cat([(shifted[k] - base[k]).reshape(-1) for k in base])
        return norm(difference) / offset_norm

    def _secant(self, anchor, base, direction, eps):
        offset, offset_norm = self._offset(direction, eps)
        shifted = {k: anchor[k] + offset[k] for k in anchor}
        step, _ = self.update_map(shifted)
        difference = torch.cat([(shifted[k] + step[k] - base[k]).reshape(-1) for k in anchor])
        return difference, norm(difference) / offset_norm

    def contractivity(self, anchor, anchor_step, directions, eps, power_iters, projector):
        base = {k: anchor[k] + anchor_step[k] for k in anchor}
        best = index = None
        for position, direction in enumerate(directions):
            probe = direction
            for _ in range(power_iters):
                secant, _ = self._secant(anchor, base, probe, eps)
                probe = self.unit_direction(secant, projector)
            _, value = self._secant(anchor, base, probe, eps)
            if best is None or value > best:
                best, index = value, position
        return best, index
