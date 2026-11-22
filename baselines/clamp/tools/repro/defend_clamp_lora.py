import argparse
import json
import math
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.func import functional_call

_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, os.path.join(_ROOT, "clamp", "generative"))
from objective import clamp_losses

from datasets import load_dataset
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from diffusers.loaders import AttnProcsLayers
from diffusers.models.attention_processor import LoRAAttnProcessor
from diffusers.optimization import get_scheduler
from torchvision import transforms
from transformers import CLIPTextModel, CLIPTokenizer

XATTN = ["to_q", "to_v", "to_k", "to_out"]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_lora_procs(unet, rank):
    procs = {}
    for name in unet.attn_processors.keys():
        cross = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
        if name.startswith("mid_block"):
            hidden = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            hidden = list(reversed(unet.config.block_out_channels))[int(name[len("up_blocks."):][0])]
        else:
            hidden = unet.config.block_out_channels[int(name[len("down_blocks."):][0])]
        procs[name] = LoRAAttnProcessor(hidden_size=hidden, cross_attention_dim=cross, rank=rank)
    return procs


def build_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pretrained_model_name_or_path", default="CompVis/stable-diffusion-v1-4")
    p.add_argument("--delta_ckpt", required=True)
    p.add_argument("--train_data_dir", required=True)
    p.add_argument("--caption_column", default="prompt")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--random_flip", action="store_true")
    p.add_argument("--center_crop", action="store_true")
    p.add_argument("--train_batch_size", type=int, default=1)
    p.add_argument("--num_train_epochs", type=int, default=50)
    p.add_argument("--max_train_samples", type=int, default=20)
    p.add_argument("--rank", type=int, default=4)
    p.add_argument("--learning_rate_lora", type=float, default=1e-4)
    p.add_argument("--learning_rate", type=float, default=1e-5)
    p.add_argument("--lr_scheduler", default="constant")
    p.add_argument("--lr_warmup_steps", type=int, default=0)
    p.add_argument("--adam_beta1", type=float, default=0.9)
    p.add_argument("--adam_beta2", type=float, default=0.999)
    p.add_argument("--adam_weight_decay", type=float, default=1e-2)
    p.add_argument("--adam_epsilon", type=float, default=1e-8)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--inner_loop_steps", type=int, default=1)
    p.add_argument("--outer_loop_steps", type=int, default=1)
    p.add_argument("--inner_steps", type=int, default=1)
    p.add_argument("--inner_lr", type=float, default=1e-2)
    p.add_argument("--lambda_long", type=float, default=1.0)
    p.add_argument("--lambda_ctr", type=float, default=2.0)
    p.add_argument("--lambda_curv", type=float, default=0.1)
    p.add_argument("--lambda_inv", type=float, default=0.5)
    p.add_argument("--long_margin", type=float, default=0.0)
    p.add_argument("--c_max", type=float, default=0.8)
    p.add_argument("--c_clip", type=float, default=0.1)
    p.add_argument("--tail_clip", type=float, default=5.0)
    p.add_argument("--kappa_min", type=float, default=3.0)
    p.add_argument("--curvature_reduction", default="mean")
    p.add_argument("--fd_eps", type=float, default=1e-1)
    p.add_argument("--curvature_delta", type=float, default=1e-1)
    p.add_argument("--subspace_rank", type=int, default=4)
    p.add_argument("--contractivity_power_iters", type=int, default=1)
    p.add_argument("--contractivity_candidates", type=int, default=3)
    p.add_argument("--max_defense_steps", type=int, default=0, help="0 = no cap (debug only)")
    p.add_argument("--calibrate_fd", default="",
                   help="comma-separated finite-difference step sizes; at the first defender "
                        "step, evaluate the probes at each and print a stability table, then "
                        "exit without training.")
    return p.parse_args()


def main():
    args = build_args()
    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda")

    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")

    print(f"[clamp-gen] loading ESD delta checkpoint {args.delta_ckpt}", flush=True)
    unet.load_state_dict(torch.load(args.delta_ckpt))

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    vae.to(device, dtype=torch.float32).eval()
    text_encoder.to(device, dtype=torch.float32).eval()

    unet.set_attn_processor(get_lora_procs(unet, args.rank))
    unet.to(device, dtype=torch.float32)
    lora_layers = AttnProcsLayers(unet.attn_processors)

    unet_attn_names, unet_attn_params = [], []
    for name, param in unet.named_parameters():
        if any(k in name for k in XATTN) and "lora" not in name:
            unet_attn_names.append(name)
            unet_attn_params.append(param)
    lora_names = [n for n, _ in unet.named_parameters() if "lora" in n]
    print(f"[clamp-gen] defended theta tensors {len(unet_attn_params)} "
          f"({sum(p.numel() for p in unet_attn_params):,} params) | "
          f"attacker LoRA tensors {len(lora_names)} "
          f"({sum(p.numel() for n, p in unet.named_parameters() if 'lora' in n):,} params)", flush=True)

    optimizer_lora = torch.optim.AdamW(lora_layers.parameters(), lr=args.learning_rate_lora,
                                       betas=(args.adam_beta1, args.adam_beta2),
                                       weight_decay=args.adam_weight_decay, eps=args.adam_epsilon)
    optimizer = torch.optim.AdamW(unet_attn_params, lr=args.learning_rate,
                                  betas=(args.adam_beta1, args.adam_beta2),
                                  weight_decay=args.adam_weight_decay, eps=args.adam_epsilon)

    dataset = load_dataset("imagefolder",
                           data_files={"train": os.path.join(args.train_data_dir, "**")},
                           drop_labels=False, drop_metadata=False)
    columns = dataset["train"].column_names
    image_column, caption_column = columns[0], args.caption_column
    n = args.max_train_samples
    dataset["train"] = (dataset["train"].shuffle(seed=args.seed).select(range(n, 2 * n))
                        if dataset["train"].shape[0] >= 2 * n
                        else dataset["train"].shuffle(seed=args.seed).select(range(n)))

    train_transforms = transforms.Compose([
        transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(args.resolution) if args.center_crop else transforms.RandomCrop(args.resolution),
        transforms.RandomHorizontalFlip() if args.random_flip else transforms.Lambda(lambda x: x),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    def preprocess(examples):
        images = [im.convert("RGB") for im in examples[image_column]]
        examples["pixel_values"] = [train_transforms(im) for im in images]
        examples["input_ids"] = tokenizer(
            [c if isinstance(c, str) else c[0] for c in examples[caption_column]],
            max_length=tokenizer.model_max_length, padding="max_length",
            truncation=True, return_tensors="pt").input_ids
        return examples

    train_dataset = dataset["train"].with_transform(preprocess)

    def collate(examples):
        return {"pixel_values": torch.stack([e["pixel_values"] for e in examples]).float(),
                "input_ids": torch.stack([e["input_ids"] for e in examples])}

    loader = torch.utils.data.DataLoader(train_dataset, shuffle=True, collate_fn=collate,
                                         batch_size=args.train_batch_size, num_workers=0)
    max_train_steps = args.num_train_epochs * len(loader)
    lr_scheduler_lora = get_scheduler(args.lr_scheduler, optimizer=optimizer_lora,
                                      num_warmup_steps=args.lr_warmup_steps,
                                      num_training_steps=max_train_steps)
    lr_scheduler = get_scheduler(args.lr_scheduler, optimizer=optimizer,
                                 num_warmup_steps=args.lr_warmup_steps,
                                 num_training_steps=max_train_steps)

    hp = {k: getattr(args, k) for k in
          ("inner_steps", "inner_lr", "lambda_long", "lambda_ctr", "lambda_curv", "lambda_inv",
           "long_margin", "c_max", "c_clip", "tail_clip", "kappa_min", "curvature_reduction",
           "fd_eps", "curvature_delta", "subspace_rank", "contractivity_power_iters",
           "contractivity_candidates")}
    json.dump(vars(args), open(os.path.join(args.output_dir, "clamp_config.json"), "w"), indent=1)

    log_path = os.path.join(args.output_dir, "clamp_train_log.csv")
    log = open(log_path, "w")
    log.write("step,epoch,kind,lora_loss,total,long,ctr,curv,inv,c_est,c_dir,B_tail,L_hat,"
              "kappa,delta_act,delta_tail,loss_harmful,sec\n")

    print(f"[clamp-gen] {len(train_dataset)} defender images | {len(loader)} steps/epoch | "
          f"{max_train_steps} total steps", flush=True)

    step = 0
    defense_steps = 0
    period = args.inner_loop_steps + args.outer_loop_steps
    t_start = time.time()
    for epoch in range(args.num_train_epochs):
        unet.train()
        for batch in loader:
            t0 = time.time()
            is_inner = (step % period) < args.inner_loop_steps
            with torch.no_grad():
                latents = vae.encode(batch["pixel_values"].to(device)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor
                encoder_hidden_states = text_encoder(batch["input_ids"].to(device))[0]
            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps,
                                      (latents.shape[0],), device=latents.device).long()
            noisy = noise_scheduler.add_noise(latents, noise, timesteps)
            target = noise

            if is_inner:
                lora_layers.requires_grad_(True)
                for p in unet_attn_params:
                    p.requires_grad_(False)
                pred = unet(noisy, timesteps, encoder_hidden_states).sample
                loss = F.mse_loss(pred.float(), target.float(), reduction="mean")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(lora_layers.parameters(), args.max_grad_norm)
                optimizer_lora.step()
                lr_scheduler_lora.step()
                optimizer_lora.zero_grad(set_to_none=True)
                log.write(f"{step},{epoch},lora,{loss.item():.8f},"
                          + ",".join([""] * 13) + f",{time.time()-t0:.2f}\n")
            else:
                lora_layers.requires_grad_(False)
                for p in unet_attn_params:
                    p.requires_grad_(True)
                phi = {n: p.detach().clone().requires_grad_(True)
                       for n, p in unet.named_parameters() if "lora" in n}

                def loss_fn(params):
                    pred = functional_call(unet, params, (noisy, timesteps, encoder_hidden_states)).sample
                    return F.mse_loss(pred.float(), target.float(), reduction="mean")

                if args.calibrate_fd:
                    print(f"\n[calibrate] one defender step, {len(phi)} LoRA tensors, "
                          f"loss={loss_fn(phi).item():.8f}", flush=True)
                    print(f"{'fd_eps':>10}{'kappa':>16}{'L_hat':>16}{'c_est':>12}"
                          f"{'delta_act':>14}{'delta_tail':>14}{'B_tail':>12}{'sec':>8}")
                    for v in [float(x) for x in args.calibrate_fd.split(",")]:
                        h = dict(hp); h["fd_eps"] = v; h["curvature_delta"] = v
                        torch.cuda.reset_peak_memory_stats()
                        tc = time.time()
                        try:
                            ls, st = clamp_losses(phi, loss_fn, h)
                            print(f"{v:>10.0e}{st['kappa'].item():>16.6g}{st['L_hat'].item():>16.6g}"
                                  f"{st['c_est'].item():>12.6f}{st['delta_act'].item():>14.6g}"
                                  f"{st['delta_tail'].item():>14.6g}{st['B_tail'].item():>12.6g}"
                                  f"{time.time()-tc:>8.1f}"
                                  f"  mem={torch.cuda.max_memory_allocated()/2**30:.1f}GiB", flush=True)
                            del ls, st
                        except Exception as e:
                            print(f"{v:>10.0e}  FAILED {type(e).__name__}: {str(e)[:70]}", flush=True)
                        torch.cuda.empty_cache()
                    print(f"[calibrate] peak GPU mem {torch.cuda.max_memory_allocated()/2**30:.1f} GiB",
                          flush=True)
                    return
                losses, stats = clamp_losses(phi, loss_fn, hp)
                total = torch.stack(list(losses.values())).sum()
                optimizer.zero_grad(set_to_none=True)
                total.backward()
                torch.nn.utils.clip_grad_norm_(unet_attn_params, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                defense_steps += 1
                log.write(
                    f"{step},{epoch},clamp,,{total.item():.8f},{losses['long'].item():.8f},"
                    f"{losses['ctr'].item():.8f},{losses.get('curv', torch.zeros(())).item():.8f},"
                    f"{losses['inv'].item():.8f},"
                    f"{stats['c_est'].item():.8f},{int(stats['c_dir'].item())},{stats['B_tail'].item():.8f},"
                    f"{stats['L_hat'].item():.8f},{stats['kappa'].item():.8f},"
                    f"{stats['delta_act'].item():.10f},{stats['delta_tail'].item():.10f},"
                    f"{stats['loss_harmful'].item():.8f},{time.time()-t0:.2f}\n")
                del losses, stats, total, phi
            log.flush()
            step += 1
            if step % 20 == 0 or (args.max_defense_steps and not is_inner):
                done = time.time() - t_start
                print(f"[clamp-gen] step {step}/{max_train_steps} epoch {epoch} "
                      f"({done/step:.2f}s/step, eta {(max_train_steps-step)*done/step/3600:.2f}h, "
                      f"peakmem {torch.cuda.max_memory_allocated()/2**30:.1f}GiB)", flush=True)
            if args.max_defense_steps and defense_steps >= args.max_defense_steps:
                print("[clamp-gen] max_defense_steps reached (debug run)", flush=True)
                step = max_train_steps
                break
        if step >= max_train_steps:
            break

    log.close()
    out = os.path.join(args.output_dir, "imma_unet_xatten_layer.pt")
    torch.save({n: p.detach().cpu() for n, p in zip(unet_attn_names, unet_attn_params)}, out)
    print(f"[clamp-gen] saved {out}  ({time.time()-t_start:.0f}s total)", flush=True)


if __name__ == "__main__":
    main()
