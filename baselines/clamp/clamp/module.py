import gc

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from timm.scheduler import CosineLRScheduler

from .losses import LabelSmoothingCrossEntropy
from .model import ImmunizedResNet
from .pcgrad import PCGrad
from .probes import TrajectoryProbes, flatten, norm

def _reduce(values, mode):
    return values.mean() if mode == "mean" else values.sum()


BENIGN_TERMS = ("primary", "adapt")
HARMFUL_TERMS = ("long", "ctr", "curv", "inv", "progress", "expected", "slope", "ratio")


class CLAMP(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters(config)
        self.config = config
        model_config = config["model"]
        self.net = ImmunizedResNet(
            num_trainable_blocks=model_config["num_trainable_blocks"],
            harmful_num_classes=config["data"]["harmful_num_classes"],
            arch=model_config.get("arch", "resnet18"),
        )
        self.criterion = LabelSmoothingCrossEntropy()
        self.hp = config["clamp"]
        self.opt_config = config["training"]
        self.use_pcgrad = bool(self.opt_config.get("pcgrad", True))
        self.automatic_optimization = False

    def forward(self, x):
        return self.net(x)

    def load_harmful_head(self, path):
        raw = torch.load(path, map_location="cpu", weights_only=False)
        state = raw.get("state_dict", raw)
        device = self.net.harmful_head.weight.device
        dtype = self.net.harmful_head.weight.dtype
        self.net.harmful_head.load_state_dict({k: v.to(device=device, dtype=dtype) for k, v in state.items()})
        return self

    def primary_loss(self, x, y):
        return self.criterion(self.net(x), y)

    def immunization_losses(self, primary_batch, harmful_batch):
        hp = self.hp
        x_primary, y_primary = primary_batch
        x_harmful, y_harmful = harmful_batch
        steps = int(hp["inner_steps"])

        probes = TrajectoryProbes(self.net, float(hp["inner_lr"]), int(hp["subspace_rank"]))
        harmful = probes.rollout(x_harmful, y_harmful, "harmful", steps)
        primary = probes.rollout(x_primary, y_primary, "primary", steps)
        project_harmful, project_primary = probes.projectors(harmful.updates, primary.updates)

        anchor_step, anchor_grad = probes.update_map(harmful.final, x_harmful, y_harmful, "harmful")
        if bool(hp.get("legacy_anchors", False)):
            anchors = harmful.anchors[1:] + [harmful.final]
            updates = harmful.updates
            tail_step, tail_grad = harmful.updates[-1], harmful.grads[-1]
        else:
            anchors = harmful.anchors + [harmful.final]
            updates = harmful.updates + [anchor_step]
            tail_step, tail_grad = anchor_step, anchor_grad
        directions = [probes.probe_direction(update, project_harmful) for update in updates]

        curvatures = [
            probes.curvature(anchor, direction, x_harmful, y_harmful, "harmful", float(hp["curvature_delta"]))
            for anchor, direction in zip(anchors, directions)
        ]
        sharpness = [
            probes.lipschitz(anchor, direction, x_harmful, y_harmful, "harmful", float(hp["fd_eps"]))
            for anchor, direction in zip(anchors, directions)
        ]
        lipschitz = sharpness[-1]

        step_flat = flatten(tail_step)
        grad_flat = flatten(tail_grad)
        candidates = [
            probes.unit_direction(step_flat, project_harmful),
            probes.unit_direction(grad_flat, project_harmful),
            probes.random_direction(project_harmful),
        ]
        contraction, choice = probes.contractivity(
            harmful.final,
            anchor_step,
            candidates,
            x_harmful,
            y_harmful,
            "harmful",
            float(hp["fd_eps"]),
            int(hp["contractivity_power_iters"]),
            project_harmful,
        )

        harmful_step = project_harmful(step_flat)
        harmful_grad = project_harmful(grad_flat)
        floor = torch.tensor(1.0 - float(hp["c_clip"]), dtype=contraction.dtype, device=contraction.device)
        budget = norm(harmful_step) / torch.maximum(1.0 - contraction, floor)
        budget = torch.clamp(budget, max=float(hp["tail_clip"]))

        loss_start = probes.loss(probes.params, x_harmful, y_harmful, "harmful")
        loss_inner = probes.loss(harmful.final, x_harmful, y_harmful, "harmful")
        delta_act = loss_start - loss_inner
        delta_tail = norm(harmful_grad) * budget + 0.5 * lipschitz * budget.pow(2)
        delta_pred = delta_act + delta_tail

        kappa_min = torch.tensor(float(hp["kappa_min"]), dtype=loss_start.dtype, device=loss_start.device)
        losses = {
            "long": float(hp["lambda_long"]) * F.relu(delta_pred - float(hp["long_margin"])),
            "ctr": float(hp["lambda_ctr"]) * F.softplus(contraction - float(hp["c_max"])),
            "curv": float(hp["lambda_curv"]) * _reduce(
                torch.stack([F.softplus(kappa_min - value) for value in curvatures]),
                hp.get("curvature_reduction", "sum"),
            ),
            "inv": -float(hp["lambda_inv"]) * loss_start,
            "primary": float(hp["lambda_primary"]) * self.primary_loss(x_primary, y_primary),
        }

        lambda_progress = float(hp.get("lambda_progress", 0.0))
        lambda_expected = float(hp.get("lambda_expected", 0.0))
        lambda_slope = float(hp.get("lambda_slope", 0.0))
        lambda_ratio = float(hp.get("lambda_ratio", 0.0))

        if lambda_progress > 0.0 or lambda_ratio > 0.0:
            losses["progress"] = lambda_progress * F.relu(delta_act)
        if lambda_expected > 0.0 or lambda_ratio > 0.0:
            simulated = [
                -flatten(grad).dot(flatten(update))
                + 0.5 * value * flatten(update).pow(2).sum()
                for grad, update, value in zip(harmful.grads, harmful.updates, sharpness)
            ]
            losses["expected"] = lambda_expected * torch.stack(simulated).sum()
        if lambda_slope > 0.0:
            losses["slope"] = lambda_slope * torch.stack(
                [norm(project_harmful(flatten(grad))) for grad in harmful.grads]
            ).mean()
        if lambda_ratio > 0.0:
            ratio = losses["progress"] / (losses["expected"].detach() + 1e-12)
            losses["ratio"] = lambda_ratio * F.softplus(ratio - 1.0)

        lambda_adapt = float(hp.get("lambda_adapt", 0.0))
        if lambda_adapt > 0.0:
            primary_directions = [
                probes.probe_direction(update, project_primary)
                for update in primary.updates
            ]
            expected = []
            for anchor, direction, grad, update in zip(
                primary.anchors, primary_directions, primary.grads, primary.updates
            ):
                value = probes.lipschitz(
                    anchor, direction, x_primary, y_primary, "primary", float(hp["fd_eps"])
                )
                inner = flatten(grad).dot(flatten(update))
                expected.append(-inner + 0.5 * value * flatten(update).pow(2).sum())
            expected_drop = torch.stack(expected).sum()
            actual_drop = probes.loss(probes.params, x_primary, y_primary, "primary") - probes.loss(
                primary.final, x_primary, y_primary, "primary"
            )
            losses["adapt"] = lambda_adapt * F.softplus(expected_drop.detach() - actual_drop)

        stats = {
            "c_est": contraction,
            "c_dir": torch.tensor(float(choice), device=contraction.device),
            "B_tail": budget,
            "L_hat": lipschitz,
            "kappa": torch.stack(curvatures).mean(),
            "delta_act": delta_act,
            "delta_tail": delta_tail,
            "loss_harmful": loss_start,
        }
        return losses, stats

    def training_step(self, batch, batch_idx):
        primary_batch, harmful_batch = batch
        dtype = next(self.parameters()).dtype
        primary_batch = (primary_batch[0].to(dtype), primary_batch[1])
        harmful_batch = (harmful_batch[0].to(dtype), harmful_batch[1])

        losses, stats = self.immunization_losses(primary_batch, harmful_batch)

        optimizer = self.optimizers()
        raw = getattr(optimizer, "optimizer", optimizer)
        raw.zero_grad(set_to_none=True)

        if self.use_pcgrad:
            groups = [
                torch.stack([losses[key] for key in BENIGN_TERMS if key in losses]).sum(),
                torch.stack([losses[key] for key in HARMFUL_TERMS if key in losses]).sum(),
            ]
            self.pcgrad.backward(groups)
            total = torch.stack([group.detach() for group in groups]).sum()
        else:
            total = torch.stack(list(losses.values())).sum()
            total.backward()
            total = total.detach()

        raw.step()
        self.scheduler.step(self.current_epoch + self.global_step / self.trainer.estimated_stepping_batches)

        self.log("train_loss", total, prog_bar=True, on_step=True, on_epoch=True)
        self.log("lr", raw.param_groups[0]["lr"], on_step=True)
        self.log_dict({f"loss/{k}": v.detach() for k, v in losses.items()}, on_step=True, on_epoch=True)
        self.log_dict({f"stat/{k}": v.detach() for k, v in stats.items()}, on_step=True, on_epoch=True)

        torch.cuda.empty_cache()
        gc.collect()
        return total

    def configure_optimizers(self):
        config = self.opt_config
        peak_lr = float(config["peak_lr"])
        optimizer = torch.optim.SGD(
            [
                {
                    "params": [p for p in self.net.feature_extractor.parameters() if p.requires_grad],
                    "lr": float(config["lr"]),
                    "weight_decay": 0.0,
                },
                {
                    "params": list(self.net.classifier.parameters()),
                    "lr": float(config["lr"]),
                    "weight_decay": float(config["classifier_weight_decay"]),
                },
            ],
            lr=peak_lr,
            momentum=float(config["momentum"]),
            nesterov=bool(config["nesterov"]),
        )
        for group in optimizer.param_groups:
            group["initial_lr"] = peak_lr
        self.scheduler = CosineLRScheduler(
            optimizer,
            t_initial=int(config["cosine_t_initial"]),
            lr_min=float(config["min_lr"]),
            warmup_t=int(config["warmup_epochs"]),
            warmup_lr_init=float(config["lr"]),
            warmup_prefix=False,
        )
        self.pcgrad = PCGrad(optimizer)
        return optimizer
