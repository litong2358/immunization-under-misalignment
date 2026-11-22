import torch
import torch.nn.functional as F

try:
    from .probes import DiffusionTrajectoryProbes, flatten, norm
except ImportError:
    from probes import DiffusionTrajectoryProbes, flatten, norm


def clamp_losses(phi, loss_fn, hp):
    probes = DiffusionTrajectoryProbes(phi, loss_fn, float(hp["inner_lr"]), int(hp["subspace_rank"]))
    harmful = probes.rollout(int(hp["inner_steps"]))
    project = probes.projector(harmful.updates)

    anchor_step, _ = probes.update_map(harmful.final)
    anchors = harmful.anchors[1:] + [harmful.final]
    updates = harmful.updates
    tail_step, tail_grad = harmful.updates[-1], harmful.grads[-1]

    directions = [probes.probe_direction(u, project) for u in updates]
    if float(hp["lambda_curv"]) == 0.0:
        curvatures = None
    else:
        curvatures = [probes.curvature(a, d, float(hp["curvature_delta"]))
                      for a, d in zip(anchors, directions)]
    sharpness = [probes.lipschitz(a, d, float(hp["fd_eps"]))
                 for a, d in zip(anchors, directions)]
    lipschitz = sharpness[-1]

    step_flat = flatten(tail_step)
    grad_flat = flatten(tail_grad)
    candidates = [
        probes.unit_direction(step_flat, project),
        probes.unit_direction(grad_flat, project),
        probes.random_direction(project),
    ][:int(hp.get("contractivity_candidates", 3))]
    contraction, choice = probes.contractivity(
        harmful.final, anchor_step, candidates,
        float(hp["fd_eps"]), int(hp["contractivity_power_iters"]), project,
    )

    harmful_step = project(step_flat)
    harmful_grad = project(grad_flat)
    floor = torch.tensor(1.0 - float(hp["c_clip"]), dtype=contraction.dtype, device=contraction.device)
    budget = norm(harmful_step) / torch.maximum(1.0 - contraction, floor)
    budget = torch.clamp(budget, max=float(hp["tail_clip"]))

    loss_start = probes.loss(probes.params)
    loss_inner = probes.loss(harmful.final)
    delta_act = loss_start - loss_inner
    delta_tail = norm(harmful_grad) * budget + 0.5 * lipschitz * budget.pow(2)
    delta_pred = delta_act + delta_tail

    kappa_min = torch.tensor(float(hp["kappa_min"]), dtype=loss_start.dtype, device=loss_start.device)
    reduction = hp.get("curvature_reduction", "mean")
    losses = {
        "long": float(hp["lambda_long"]) * F.relu(delta_pred - float(hp["long_margin"])),
        "ctr": float(hp["lambda_ctr"]) * F.softplus(contraction - float(hp["c_max"])),
        "inv": -float(hp["lambda_inv"]) * loss_start,
    }
    if curvatures is not None:
        curv_stack = torch.stack([F.softplus(kappa_min - v) for v in curvatures])
        losses["curv"] = float(hp["lambda_curv"]) * (
            curv_stack.mean() if reduction == "mean" else curv_stack.sum())
    stats = {
        "c_est": contraction.detach(),
        "c_dir": torch.tensor(float(choice)),
        "B_tail": budget.detach(),
        "L_hat": lipschitz.detach(),
        "kappa": (torch.stack(curvatures).mean().detach() if curvatures is not None
                  else torch.tensor(float("nan"))),
        "delta_act": delta_act.detach(),
        "delta_tail": delta_tail.detach(),
        "loss_harmful": loss_start.detach(),
    }
    return losses, stats
