import math

import torch
import torch.nn.functional as F

INTERP = {"mode": "bilinear"}
S = 256
OUT_SIDE = 224


def sample_views(B, gen, device):
    area = S * S
    scale = torch.empty(B, 10, device=device).uniform_(0.08, 1.0, generator=gen) * area
    logr = torch.empty(B, 10, device=device).uniform_(math.log(3 / 4), math.log(4 / 3), generator=gen)
    r = torch.exp(logr)
    w = torch.round(torch.sqrt(scale * r))
    h = torch.round(torch.sqrt(scale / r))
    ok = (w <= S) & (h <= S) & (w > 0) & (h > 0)
    first = torch.where(ok.any(1), ok.float().argmax(1), torch.full((B,), -1, device=device))
    idx = first.clamp_min(0)
    w = torch.where(first >= 0, w.gather(1, idx[:, None])[:, 0], torch.full((B,), float(S), device=device))
    h = torch.where(first >= 0, h.gather(1, idx[:, None])[:, 0], torch.full((B,), float(S), device=device))
    i0 = torch.floor(torch.rand(B, device=device, generator=gen) * (S - h + 1))
    j0 = torch.floor(torch.rand(B, device=device, generator=gen) * (S - w + 1))
    flip = torch.rand(B, device=device, generator=gen) < 0.5
    jit = torch.empty(B, 3, device=device).uniform_(0.6, 1.4, generator=gen)
    order = torch.randperm(3, generator=gen, device=device)
    return {"i": i0, "j": j0, "h": h, "w": w, "flip": flip, "jit": jit, "order": order}


def gray(x):
    return (0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3])


def apply_view(x01, v):
    B = x01.size(0)
    sx = v["w"] / S
    sy = v["h"] / S
    cx = (v["j"] + v["w"] / 2) / S * 2 - 1
    cy = (v["i"] + v["h"] / 2) / S * 2 - 1
    sxf = torch.where(v["flip"], -sx, sx)
    theta = torch.zeros(B, 2, 3, device=x01.device, dtype=x01.dtype)
    theta[:, 0, 0], theta[:, 0, 2] = sxf, cx
    theta[:, 1, 1], theta[:, 1, 2] = sy, cy
    grid = F.affine_grid(theta, (B, 3, OUT_SIDE, OUT_SIDE), align_corners=False)
    y = F.grid_sample(x01, grid, mode=INTERP["mode"], padding_mode="border", align_corners=False)
    if INTERP["mode"] == "bicubic":
        y = y.clamp(0, 1)
    for k in v["order"].tolist():
        f = v["jit"][:, k].view(B, 1, 1, 1)
        if k == 0:
            y = (y * f).clamp(0, 1)
        elif k == 1:
            m = gray(y).mean(dim=(1, 2, 3), keepdim=True)
            y = ((y - m) * f + m).clamp(0, 1)
        else:
            g = gray(y)
            y = ((y - g) * f + g).clamp(0, 1)
    return y


def center_view(x01):
    o = (S - OUT_SIDE) // 2
    return x01[:, :, o:o + OUT_SIDE, o:o + OUT_SIDE]


def group_topk_mask(score, rho):
    B, _, H, W = score.shape
    per_px = score.pow(2).sum(1).view(B, -1)
    k = max(1, int(round(rho * H * W)))
    idx = per_px.topk(k, dim=1).indices
    m = torch.zeros_like(per_px)
    m.scatter_(1, idx, 1.0)
    return m.view(B, 1, H, W)
