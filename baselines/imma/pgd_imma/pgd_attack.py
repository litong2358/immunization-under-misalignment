from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import pathlib
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer

_IMMA_ROOT = os.environ.get("IMMA_ROOT")
REPO = pathlib.Path(_IMMA_ROOT) if _IMMA_ROOT else Path(__file__).resolve().parents[2]
MODEL_NAME = "CompVis/stable-diffusion-v1-4"


def build_unets(delta_ckpt: Path, imma_ckpt: Path, device, dtype):
    m0 = UNet2DConditionModel.from_pretrained(MODEL_NAME, subfolder="unet")
    m0.load_state_dict(torch.load(delta_ckpt, map_location="cpu"))

    imma = torch.load(imma_ckpt, map_location="cpu")
    m1 = UNet2DConditionModel.from_pretrained(MODEL_NAME, subfolder="unet")
    sd = m1.state_dict()
    sd.update(torch.load(delta_ckpt, map_location="cpu"))
    base = {k: v.clone() for k, v in sd.items() if k in imma}
    sd.update(imma)
    m1.load_state_dict(sd)

    report = assert_m1(base, imma)
    for m in (m0, m1):
        m.requires_grad_(False)
        m.to(device, dtype=dtype)
        m.eval()
    return m0, m1, report


def assert_m1(base: dict, imma: dict) -> dict:
    missing = [k for k in imma if k not in base]
    if missing:
        raise RuntimeError(f"imma_ckpt has {len(missing)} keys absent from the UNet, e.g. {missing[:3]}")

    changed = [k for k, v in imma.items() if not torch.equal(base[k].float(), v.float())]
    if not changed:
        raise RuntimeError(
            "imma_ckpt is bitwise identical to M0 on every key -- M1 == M0, so there is "
            "nothing to attack and any 'PGD on M1' claim would be false."
        )
    non_attn = [k for k in imma if ("attn1" not in k and "attn2" not in k)]
    if non_attn:
        raise RuntimeError(f"imma_ckpt touches non-attention params: {non_attn[:3]}")

    n1 = sum(1 for k in changed if "attn1" in k)
    n2 = sum(1 for k in changed if "attn2" in k)
    delta_norm = sum((imma[k].float() - base[k].float()).norm().item() ** 2 for k in changed) ** 0.5
    print(f"[M1 gate/structural] {len(changed)}/{len(imma)} IMMA tensors differ from M0 "
          f"(self-attn {n1}, cross-attn {n2}); ||theta1 - theta0||_2 = {delta_norm:.4f}")
    return {"n_imma_tensors": len(imma), "n_changed": len(changed),
            "n_attn1": n1, "n_attn2": n2, "weight_delta_l2": delta_norm}


def assert_m1_behavioural(m0, m1, latents, noise, t, cond) -> dict:
    with torch.no_grad():
        zt = SCHED.add_noise(latents, noise, t)
        l0 = F.mse_loss(m0(zt, t, cond).sample.float(), noise.float()).item()
        l1 = F.mse_loss(m1(zt, t, cond).sample.float(), noise.float()).item()
    if abs(l0 - l1) < 1e-8:
        raise RuntimeError(
            f"M0 and M1 give an identical loss ({l0:.8f}) on the same batch -- the IMMA "
            "weights are not in the forward pass. Aborting rather than attacking M0."
        )
    print(f"[M1 gate/behavioural] diffusion loss on the same batch: "
          f"M0 {l0:.5f}  vs  M1 {l1:.5f}  (IMMA raises it by {(l1 - l0) / l0 * 100:+.1f}%)")
    return {"loss_m0": l0, "loss_m1": l1}


def encode(vae, x01: torch.Tensor) -> torch.Tensor:
    return vae.encode(x01 * 2.0 - 1.0).latent_dist.mean * vae.config.scaling_factor


def eot_draws(n_eot: int, bsz: int, device, generator=None):
    T = SCHED.config.num_train_timesteps
    span = T // n_eot
    draws = []
    for k in range(n_eot):
        t = torch.randint(k * span, (k + 1) * span, (bsz,), device=device, generator=generator).long()
        flip = bool(torch.rand(1, generator=generator, device=device).item() < 0.5)
        draws.append((t, flip))
    return draws


def diffusion_loss(unet, vae, x01, cond, t, flip, noise):
    xin = torch.flip(x01, dims=[3]) if flip else x01
    z = encode(vae, xin)
    zt = SCHED.add_noise(z, noise, t)
    return F.mse_loss(unet(zt, t, cond).sample.float(), noise.float())


def pgd(args, unet, vae, x01, cond):
    eps, alpha = args.eps / 255.0, args.alpha / 255.0
    device = x01.device

    g_meas = torch.Generator(device=device).manual_seed(args.seed)
    meas = [(t, flip, None) for t, flip in eot_draws(args.n_eot, x01.shape[0], device, g_meas)]
    meas = [(t, flip, torch.randn(x01.shape[0], 4, x01.shape[2] // 8, x01.shape[3] // 8,
                                  device=device, generator=g_meas, dtype=torch.float32))
            for t, flip, _ in meas]

    def measure(x):
        with torch.no_grad():
            return float(sum(diffusion_loss(unet, vae, x, cond, t, flip, n).item()
                             for t, flip, n in meas) / len(meas))

    l_before = measure(x01)

    if args.loss == "random":
        if args.rand_form == "rademacher":
            delta = torch.sign(torch.randn_like(x01)) * eps
        else:
            delta = (torch.rand_like(x01) * 2 - 1) * eps
        delta = ((x01 + delta).clamp(0, 1) - x01)
        return delta, l_before, measure((x01 + delta).clamp(0, 1))

    delta = ((torch.rand_like(x01) * 2 - 1) * eps).requires_grad_(True)
    direction = 1.0 if args.direction == "min" else -1.0

    for step in range(args.steps):
        x_adv = (x01 + delta).clamp(0, 1)
        draws = eot_draws(args.n_eot, x01.shape[0], device)
        noises = [torch.randn(x01.shape[0], 4, x01.shape[2] // 8, x01.shape[3] // 8,
                              device=device, dtype=torch.float32) for _ in draws]

        obj = sum(diffusion_loss(unet, vae, x_adv, cond, t, flip, n)
                  for (t, flip), n in zip(draws, noises)) / len(draws)
        grad, = torch.autograd.grad(obj, delta)

        with torch.no_grad():
            delta -= direction * alpha * grad.sign()
            delta.clamp_(-eps, eps)
            delta.copy_((x01 + delta).clamp(0, 1) - x01)

        if args.verbose and (step + 1) % 10 == 0:
            print(f"      step {step + 1:3d}/{args.steps}  obj={float(obj):.6f}  "
                  f"L1={measure((x01 + delta).clamp(0, 1)):.6f}")

    delta = delta.detach()
    return delta, l_before, measure((x01 + delta).clamp(0, 1))


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--loss", required=True, choices=["l1", "random"])
    ap.add_argument("--direction", choices=["min", "max"], default="max", help="l1 only")
    ap.add_argument("--tag", required=True, help="output dataset name: data/<artist>_pgd_<tag>")
    ap.add_argument("--artist", default="vangogh")
    ap.add_argument("--prompt", default="An artwork by Van Gogh")
    ap.add_argument("--imma-ckpt", default="runs_repro/vangogh/imma.pt")
    ap.add_argument("--delta-ckpt", default="diffusers-VanGogh-ESDx1-UNET.pt")
    ap.add_argument("--eps", type=float, default=4.0, help="L_inf budget in /255")
    ap.add_argument("--alpha", type=float, default=0.4, help="PGD step in /255")
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--rand-form", choices=["uniform", "rademacher"], default="uniform",
                    help="--loss random only")
    ap.add_argument("--n-eot", type=int, default=4, help="stratified timestep samples per step")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--t-max", type=int, default=1000, help="restrict timesteps to [0, t_max)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None, help="perturb only the first N images")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda")
    torch.manual_seed(args.seed)

    global SCHED
    SCHED = DDPMScheduler.from_pretrained(MODEL_NAME, subfolder="scheduler")
    if args.t_max != SCHED.config.num_train_timesteps:
        SCHED.config.num_train_timesteps = args.t_max
        print(f"[cfg] timesteps restricted to [0, {args.t_max})")

    print("[load] SD-v1.4 + ESD + IMMA ...")
    vae = AutoencoderKL.from_pretrained(MODEL_NAME, subfolder="vae").to(device, torch.float32).eval()
    vae.requires_grad_(False)
    tok = CLIPTokenizer.from_pretrained(MODEL_NAME, subfolder="tokenizer")
    txt = CLIPTextModel.from_pretrained(MODEL_NAME, subfolder="text_encoder").to(device, torch.float32).eval()
    txt.requires_grad_(False)

    m0, m1, gate = build_unets(REPO / args.delta_ckpt, REPO / args.imma_ckpt, device, torch.float32)
    unet = m1

    with torch.no_grad():
        ids = tok([args.prompt], padding="max_length", max_length=tok.model_max_length,
                  truncation=True, return_tensors="pt").input_ids.to(device)
        cond1 = txt(ids)[0]

    src = REPO / "data" / args.artist
    all_files = sorted(src.glob("*.png"))
    out = REPO / "data" / f"{args.artist}_pgd_{args.tag}"
    out.mkdir(parents=True, exist_ok=True)

    files = all_files
    if args.limit:
        files = files[:args.limit]
    print(f"[cfg] {len(files)} images  eps={args.eps}/255  alpha={args.alpha}/255  "
          f"steps={args.steps}  n_eot={args.n_eot}  -> {out.relative_to(REPO)}")

    import numpy as np
    x0 = torch.stack([torch.from_numpy(
        np.asarray(Image.open(f).convert("RGB"), dtype="float32") / 255.0
    ).permute(2, 0, 1) for f in files[:2]]).to(device)
    with torch.no_grad():
        z = encode(vae, x0)
        n = torch.randn_like(z)
        t = torch.full((z.shape[0],), 500, device=device, dtype=torch.long)
        beh = assert_m1_behavioural(m0, m1, z, n, t, cond1.expand(z.shape[0], -1, -1))

    rows, t_start = [], time.time()
    for i in range(0, len(files), args.batch):
        chunk = files[i:i + args.batch]
        x01 = torch.stack([torch.from_numpy(
            np.asarray(Image.open(f).convert("RGB"), dtype="float32") / 255.0
        ).permute(2, 0, 1) for f in chunk]).to(device)
        cond = cond1.expand(x01.shape[0], -1, -1)

        delta, lb, la = pgd(args, unet, vae, x01, cond)
        x_adv = (x01 + delta).clamp(0, 1)

        x_u8 = (x_adv * 255).round()
        x_q = x_u8 / 255.0
        g = torch.Generator(device=device).manual_seed(args.seed)
        meas = [(t, f_, torch.randn(x01.shape[0], 4, 64, 64, device=device, generator=g))
                for t, f_ in eot_draws(args.n_eot, x01.shape[0], device, g)]
        with torch.no_grad():
            lq = float(sum(diffusion_loss(unet, vae, x_q, cond, t, f_, n).item()
                           for t, f_, n in meas) / len(meas))

        for j, f in enumerate(chunk):
            arr = x_u8[j].permute(1, 2, 0).to(torch.uint8).cpu().numpy()
            Image.fromarray(arr).save(out / f.name)
            d = (x_adv[j] - x01[j])
            rows.append({"name": f.name, "loss_before": lb, "loss_after": la,
                         "loss_after_uint8": lq,
                         "linf_255": float(d.abs().max() * 255),
                         "l2": float(d.norm()),
                         "sha256": sha256_file(out / f.name)})
        done = min(i + args.batch, len(files))
        el = time.time() - t_start
        print(f"  [{done:3d}/{len(files)}] L1 {lb:.5f} -> {la:.5f} (uint8 {lq:.5f})  "
              f"{el / done * len(files) / 60:.1f} min total est.")

    with open(out / "metadata.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["file_name", "prompt"])
        w.writeheader()
        for p in sorted(out.glob("*.png")):
            w.writerow({"file_name": p.name, "prompt": args.prompt})

    moved = sum(1 for r in rows if r["linf_255"] > 0.5)
    mean_b = sum(r["loss_before"] for r in rows) / len(rows)
    mean_a = sum(r["loss_after"] for r in rows) / len(rows)
    mean_q = sum(r["loss_after_uint8"] for r in rows) / len(rows)
    print(f"\n[result] mean L1  before {mean_b:.5f}  after {mean_a:.5f}  "
          f"after-uint8 {mean_q:.5f}   ({mean_a / mean_b - 1:+.1%})")
    print(f"[result] {moved}/{len(rows)} images actually moved")
    if moved == 0:
        raise RuntimeError("PGD changed nothing -- broken gradient path, not a null result.")
    if abs(mean_q - mean_a) > 0.25 * abs(mean_a - mean_b) and abs(mean_a - mean_b) > 1e-6:
        print("[warn] uint8 quantization erased >25% of the loss change; the effect may "
              "not survive saving to PNG.")

    manifest = {
        "loss": args.loss,
        "direction": {"l1": args.direction, "random": "none"}[args.loss],
        "batch": args.batch,
        "eps_over_255": args.eps, "alpha_over_255": args.alpha, "steps": args.steps,
        "rand_form": args.rand_form if args.loss == "random" else None,
        "n_eot": args.n_eot, "t_max": args.t_max, "seed": args.seed,
        "prompt": args.prompt, "source_dir": f"data/{args.artist}",
        "imma_ckpt": args.imma_ckpt,
        "m1_gate_structural": gate, "m1_gate_behavioural": beh,
        "mean_loss_before": mean_b, "mean_loss_after": mean_a,
        "mean_loss_after_uint8": mean_q, "n_images_moved": moved,
        "n_perturbed": len(rows), "n_images_total": len(list(out.glob("*.png"))),
        "files": rows,
    }
    with open(out / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[ok] wrote {out.relative_to(REPO)}  ({len(rows)} images + metadata.csv + manifest.json)")


if __name__ == "__main__":
    main()
