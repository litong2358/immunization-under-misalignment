import argparse
import copy
import json
import math
import os
import sys
import time

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from PIL import Image
from torchvision import datasets, transforms

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "..", "cn"))
from clamp.model import ImmunizedResNet
from cn_to_clamp import load_theta_I
import views as pe

MEAN = (0.48145466, 0.4578275, 0.40821073)
STD = (0.26862954, 0.26130258, 0.27577711)
C = 196
D = 512


def bn_layers(m):
    return [x for x in m.modules() if isinstance(x, torch.nn.modules.batchnorm._BatchNorm)]


def log_pr_group(phi):
    Cm = phi.t() @ phi
    tr = torch.diagonal(Cm).sum()
    return (tr * tr / ((Cm * Cm).sum() + 1e-12)).log()


def classmean_logpr(Fk, Y, C):
    n_c = torch.bincount(Y, minlength=C).to(Fk.dtype)
    mu_c = torch.zeros(C, Fk.size(1), dtype=Fk.dtype, device=Fk.device).index_add_(0, Y, Fk)
    mu_c = mu_c / n_c.clamp_min(1)[:, None]
    present = n_c > 0
    dev = (mu_c - Fk.mean(0))[present] * n_c[present].sqrt()[:, None]
    G = dev @ dev.t()
    tr = torch.diagonal(G).sum()
    return (tr * tr / ((G * G).sum() + 1e-12)).log()


def feat_stats(H, Y, C):
    with torch.no_grad():
        H = H.double()
        n_c = torch.bincount(Y, minlength=C).double()
        mu_c = torch.zeros(C, H.size(1), dtype=H.dtype, device=H.device).index_add_(0, Y, H)
        mu_c = mu_c / n_c.clamp_min(1)[:, None]
        mu = H.mean(0)
        sb = (n_c[:, None] * (mu_c - mu).pow(2)).sum()
        sw = (H - mu_c[Y]).pow(2).sum()
        return {"logpr": log_pr_group(H).item(), "logpr_centered": log_pr_group(H - mu).item(),
                "classmean_logpr": classmean_logpr(H, Y, C).item(),
                "fisher_ratio": (sb / sw).item(),
                "trM": ((H - mu).pow(2).sum() / H.size(0)).item(), "trSB": (sb / H.size(0)).item()}


def holdout_score(Hho, yho):
    def f(W, b):
        lo = Hho @ W.t() + b
        return F.cross_entropy(lo, yho), 100.0 * (lo.argmax(1) == yho).float().mean().item()
    return f


def simulate(Ftr, ytr, Hho, yho, a, gen, need_graph, score=None):
    score = score or holdout_score(Hho, yho)
    V, n, _ = Ftr.shape
    dev = Ftr.device
    bound = 1.0 / math.sqrt(D)
    W = ((torch.rand(C, D, device=dev, generator=gen) * 2 - 1) * bound).to(Ftr.dtype)
    b = ((torch.rand(C, device=dev, generator=gen) * 2 - 1) * bound).to(Ftr.dtype)
    mW = mb = None
    Yoh = F.one_hot(ytr, C).to(Ftr.dtype)
    per_ep = n // a.batch
    ck = set(a.ckpt_steps)
    T = max(a.ckpt_steps)
    ces, accs = [], []
    t = 0
    ctx = torch.enable_grad() if need_graph else torch.no_grad()
    with ctx:
        while t < T:
            perm = torch.randperm(n, device=dev, generator=gen)
            for i in range(per_ep):
                idx = perm[i * a.batch:(i + 1) * a.batch]
                vk = torch.randint(V, (a.batch,), device=dev, generator=gen)
                Fb = Ftr[vk, idx]
                E = (torch.softmax(Fb @ W.t() + b, 1) - Yoh[idx]) / a.batch
                dW = E.t() @ Fb + a.wd * W
                db = E.sum(0) + a.wd * b
                mW = dW if mW is None else a.mom * mW + dW
                mb = db if mb is None else a.mom * mb + db
                W = W - a.lr * mW
                b = b - a.lr * mb
                t += 1
                if t in ck:
                    lv, mv = score(W, b)
                    ces.append(lv)
                    accs.append(mv)
                if t >= T:
                    break
    return torch.stack(ces).mean(), accs


def save_set(a, ds, X, D8, meta):
    meta["l0_frac_pixels_moved"] = (D8 != 0).any(1).float().mean().item()
    meta["linf_over_255_max"] = int(D8.abs().max().item())
    for (path, _), i in zip(ds.samples, range(len(ds))):
        cls = os.path.basename(os.path.dirname(path))
        stem = os.path.splitext(os.path.basename(path))[0]
        d = os.path.join(a.out, a.train_split, cls)
        os.makedirs(d, exist_ok=True)
        x = (X[i].to(torch.int16) + D8[i].to(torch.int16)).clamp(0, 255).to(torch.uint8).cpu()
        Image.fromarray(x.permute(1, 2, 0).numpy()).save(os.path.join(d, stem + ".png"))
    link = os.path.join(a.out, a.test_split)
    if a.test_src and not os.path.exists(link):
        os.symlink(a.test_src, link)
    json.dump(meta, open(os.path.join(a.out, "meta.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in meta.items() if k != "J_hist"}, indent=1))
    print("saved", a.out)


def prep(a):
    sz = 224 if a.cn_view else pe.S
    ds = datasets.ImageFolder(a.src_train, transforms.Compose([transforms.Resize((sz, sz)),
                                                                transforms.ToTensor()]))
    X = torch.stack([torch.round(ds[i][0] * 255).to(torch.uint8) for i in range(len(ds))])
    save_set(a, ds, X, torch.zeros_like(X, dtype=torch.int8),
             {"mode": "prep", "src_train": a.src_train, "N": len(ds), "C": len(ds.classes)})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", required=True, choices=["sim", "attack", "rand", "prep"],
                   help="rand = random +/-eps on the perturbed pixels (same-budget, same-support "
                        "control, see --sparse_rho); "
                        "prep = write --src_train resized to PNG as the clean set")
    p.add_argument("--ckpt", default="", help="theta_I; empty = theta_0 (timm resnet18)")
    p.add_argument("--ckpt_format", default="clamp", choices=["clamp", "cn"])
    p.add_argument("--clean_root", required=True)
    p.add_argument("--out", default="")
    p.add_argument("--eps", type=float, default=None, help="/255; default 2, or 1 with --cn_view")
    p.add_argument("--alpha", type=float, default=0.25, help="PGD step as a fraction of eps")
    p.add_argument("--steps", type=int, default=40, help="PGD steps")
    p.add_argument("--views", type=int, default=4, help="training views per image per PGD step")
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--mom", type=float, default=0.9)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--ckpt_steps", default="auto",
                   help="SGD steps of the real adaptation's epoch 10..50 checkpoints; "
                        "'auto' = ep x (N // batch) for ep in 10..50")
    p.add_argument("--train_split", default="train", help="train folder name under the roots")
    p.add_argument("--test_split", default="test", help="test folder name under the roots")
    p.add_argument("--test_src", default="",
                   help="original test images; symlinked as <out>/<test_split>")
    p.add_argument("--src_train", default="", help="prep: original train ImageFolder")
    p.add_argument("--bn_passes", type=int, default=2,
                   help="augmented passes over the clean set to estimate eval-mode BN statistics")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--objective", default="cfprobe", choices=["cfprobe", "holdout"],
                   help="cfprobe = closed-form simulation of the attacker's own probe protocol "
                        "(the paper's attack); holdout = the fully unrolled two-fold simulation "
                        "(exact but much slower; used as a fidelity reference)")
    p.add_argument("--cf_lrs", default="0.1,0.2,0.3,0.5,1,2,3,5,10",
                   help="cfprobe, --cn_view only: the probe's constant-lr grid; the simulated "
                        "protocol keeps the lr with the best mean soft held-out accuracy "
                        "(lrs with eta*kappa*lam_max/C > 2 are 'diverged')")
    p.add_argument("--cf_curv", type=float, default=3.0,
                   help="kappa, multiplier on the uniform-softmax cross-entropy curvature 1/C "
                        "of the linearised probe dynamics")
    p.add_argument("--cf_probe_batch", type=int, default=0,
                   help="the real probe's batch size (steps per epoch = ceil(N/b) on --cn_view, "
                        "N//b otherwise); 0 = --batch")
    p.add_argument("--cf_inits", type=int, default=1,
                   help="number of nn.Linear-law head initialisations averaged (seed+555..)")
    p.add_argument("--cf_folds", type=int, default=5,
                   help="K-fold cross-fit over the attacker's training images: fit the head on the "
                        "perturbed features of K-1 folds, score it on the clean, un-augmented "
                        "features of the held-out fold")
    p.add_argument("--cf_temp", type=float, default=0.25,
                   help="temperature factor of the differentiable held-out score "
                        "softmax(Z / (temp * std Z))[y], times the detached std of the held-out logits")
    p.add_argument("--logpr_beta", type=float, default=1.0,
                   help="weight of the log participation ratio term (per (view, batch) group of "
                        "the extraction plan, averaged), maximized alongside the cfprobe score. "
                        "0 = off.")
    p.add_argument("--lp_equal_vote", action=argparse.BooleanOptionalAction, default=True,
                   help="renormalize the logPR gradient's L1 to match the cfprobe gradient's L1 "
                        "before mixing, so beta is the exact pixel-vote ratio on both the CLAMP "
                        "(dense view) and CN (--cn_view) paths. Requires --logpr_beta != 0.")
    p.add_argument("--cn_view", action="store_true",
                   help="CN native path: 224x224 canvas, a single deterministic view (no random "
                        "crop -- the Resize((224,224)) is done at prep), eval-mode BN with the "
                        "released running statistics everywhere.")
    p.add_argument("--sparse_rho", type=float, default=None,
                   help="support of the perturbation, as a fraction of the pixels of each image. "
                        "Unset = every pixel, projected steps of size --alpha. A value in (0, 1) = every "
                        "step re-selects the pixels with the largest gradient (the 3 channels as a unit) "
                        "and sets them to +/-eps, all others to zero.")
    a = p.parse_args()
    if a.eps is None:
        a.eps = 1.0 if a.cn_view else 2.0
    a.sparse_rho = a.sparse_rho or 0.0
    assert 0.0 <= a.sparse_rho <= 1.0, "--sparse_rho is a fraction in (0, 1]"
    if a.logpr_beta != 0:
        assert a.objective == "cfprobe", "--logpr_beta requires --objective cfprobe"
    if a.lp_equal_vote:
        assert a.logpr_beta != 0, "--lp_equal_vote requires --logpr_beta != 0"
    if a.cn_view:
        a.views = 1
        pe.sample_views = lambda *s, **k: None
        pe.apply_view = lambda x, v: x
        pe.center_view = lambda x: x
    if a.mode == "prep":
        prep(a)
        return
    dev = "cuda"
    torch.manual_seed(a.seed)
    gen = torch.Generator(device=dev).manual_seed(a.seed)
    mt = torch.tensor(MEAN, device=dev).view(1, 3, 1, 1)
    st = torch.tensor(STD, device=dev).view(1, 3, 1, 1)

    global C
    ds = datasets.ImageFolder(os.path.join(a.clean_root, a.train_split), transforms.ToTensor())
    C = len(ds.classes)
    N = len(ds)
    X = torch.stack([torch.round(ds[i][0] * 255).to(torch.uint8) for i in range(N)]).to(dev)
    Y = torch.tensor([t for _, t in ds.samples], device=dev)
    a.ckpt_steps = ([e * (N // a.batch) for e in (10, 20, 30, 40, 50)] if a.ckpt_steps == "auto"
                    else [int(v) for v in a.ckpt_steps.split(",")])
    if a.mode == "rand":
        eps = a.eps / 255.0
        D8 = torch.empty_like(X, dtype=torch.int8)
        for k in range(0, N, 256):
            x01 = X[k:k + 256].float() / 255
            sgn = torch.where(torch.rand(x01.shape, device=dev, generator=gen) < 0.5, -1.0, 1.0)
            if a.sparse_rho > 0:
                B, _, Hh, Ww = x01.shape
                r = torch.rand(B, Hh * Ww, device=dev, generator=gen)
                idx = r.topk(max(1, int(round(a.sparse_rho * Hh * Ww))), dim=1).indices
                sgn = sgn * torch.zeros_like(r).scatter_(1, idx, 1.0).view(B, 1, Hh, Ww)
            D8[k:k + 256] = (torch.round((x01 + sgn * eps).clamp(0, 1) * 255) - X[k:k + 256].float()).to(torch.int8)
        save_set(a, ds, X, D8, {"mode": "rand", "eps": a.eps, "sparse_rho": a.sparse_rho, "seed": a.seed,
                                "N": N, "C": C})
        return

    net = ImmunizedResNet(num_trainable_blocks=2, harmful_num_classes=C).to(dev).float()
    if a.ckpt:
        load_theta_I(net, a.ckpt, a.ckpt_format)
    fe = net.feature_extractor
    for q in fe.parameters():
        q.requires_grad_(False)

    fe_ev = copy.deepcopy(fe)
    if not a.cn_view:
        for m in bn_layers(fe_ev):
            m.reset_running_stats()
            m.momentum = None
        fe_ev.train()
    with torch.no_grad():
        for _ in range(0 if a.cn_view else a.bn_passes):
            perm = torch.randperm(N, device=dev, generator=gen)
            for k in range(0, N - a.batch + 1, a.batch):
                bi = perm[k:k + a.batch]
                v = pe.sample_views(len(bi), gen, dev)
                fe_ev((pe.apply_view(X[bi].float() / 255, v) - mt) / st)
        fe_ev.eval()
        H = torch.cat([fe_ev((pe.center_view(X[k:k + 256].float() / 255) - mt) / st)
                       for k in range(0, N, 256)])
    del fe_ev

    for m in bn_layers(fe):
        m.momentum = 0.0
    (fe.eval() if a.cn_view else fe.train())
    run_state = {k: v.clone() for k, v in fe.state_dict().items() if "running" in k}
    feat = lambda x224: fe((x224 - mt) / st)

    def extract(delta):
        Fv = torch.empty(a.views, N, D, device=dev)
        plan = []
        with torch.no_grad():
            for k in range(a.views):
                perm = torch.randperm(N, device=dev, generator=gen)
                for j in range(0, N, a.batch):
                    bi = perm[j:j + a.batch]
                    v = pe.sample_views(len(bi), gen, dev)
                    x = X[bi].float() / 255
                    if delta is not None:
                        x = (x + delta[bi]).clamp(0, 1)
                    Fv[k, bi] = feat(pe.apply_view(x, v))
                    plan.append((k, bi, v))
        return Fv, plan

    lastobj = {}

    cf, cf_meta = None, {}
    if a.objective == "cfprobe":
        gf = torch.Generator(device=dev).manual_seed(a.seed + 4242)
        cf_K = max(2, a.cf_folds)
        cf_fold = torch.randperm(N, device=dev, generator=gf) % cf_K
        bound = 1.0 / math.sqrt(D)
        W0s = []
        for i in range(max(1, a.cf_inits)):
            g0 = torch.Generator(device="cpu").manual_seed(a.seed + 555 + i)
            W0 = (torch.rand(C, D, generator=g0) * 2 - 1) * bound
            b0 = (torch.rand(C, generator=g0) * 2 - 1) * bound
            W0s.append(torch.cat([W0, b0[:, None]], 1).to(dev))
        H1 = torch.cat([H, torch.ones(N, 1, device=dev)], 1)
        pb = a.cf_probe_batch or a.batch
        if a.cn_view:
            per_ep = math.ceil(N / pb)
            cf_ckpt = [e * per_ep for e in (10, 20, 30, 40, 50)]
            cf_sched = {float(v): [float(v)] * max(cf_ckpt) for v in a.cf_lrs.split(",")}
        else:
            per_ep = N // pb
            cf_ckpt = list(a.ckpt_steps)
            lr_ep = [0.5 * a.lr * (1 + math.cos(math.pi * (ep - 1) / 50)) for ep in range(1, 51)]
            cf_sched = {None: [lr_ep[min((t - 1) // per_ep, 49)] for t in range(1, max(cf_ckpt) + 1)]}
        cf = dict(fold=cf_fold, K=cf_K, W0s=W0s, H1=H1, ckpt=cf_ckpt, sched=cf_sched)
        cf_meta = {"cf_ckpt_steps": cf_ckpt, "cf_lrs": sorted(k for k in cf_sched if k is not None) or [a.lr],
                   "cf_beta": a.mom, "cf_wd": a.wd, "cf_curv": a.cf_curv, "cf_probe_batch": pb,
                   "cf_inits": len(W0s), "cf_folds": cf_K, "cf_fit_size": int((cf_fold != 0).sum().item()),
                   "cf_temp": a.cf_temp}
        print(f"[cf] ckpt {cf_ckpt}  beta {a.mom} wd {a.wd} kappa {a.cf_curv}  folds {cf_K} "
              f"(fit {cf_meta['cf_fit_size']})  inits {len(W0s)}  temp {a.cf_temp:g}  "
              f"lrs {cf_meta['cf_lrs']}", flush=True)

    def cf_segment(W, V, M, S, lrs, t0, t1, beta, wd, kappa):
        for t in range(t0, t1):
            Wc = W - W.mean(0, keepdim=True)
            d = (kappa / C) * (Wc @ M) - S + wd * W
            V = beta * V + d
            W = W - lrs[t] * V
        return W, V

    def cf_run(M, S, lrs, ckpts, beta, wd, kappa, W0, need_grad):
        W, V = W0, torch.zeros_like(W0)
        outs, t0 = [], 0
        for t1 in ckpts:
            if need_grad:
                W, V = torch.utils.checkpoint.checkpoint(cf_segment, W, V, M, S, lrs, t0, t1, beta, wd, kappa,
                                                         use_reentrant=False)
            else:
                W, V = cf_segment(W, V, M, S, lrs, t0, t1, beta, wd, kappa)
            outs.append(W)
            t0 = t1
        return outs

    def cf_score(Wt, Hb, yb):
        Z = Hb @ Wt.t()
        Ts = Z.detach().std().clamp_min(1e-6)
        soft = torch.softmax(Z / (Ts * a.cf_temp), 1).gather(1, yb[:, None]).mean()
        return soft, (Z.argmax(1) == yb).float().mean().item()

    def cf_objective(Fv, need_grad, plan=None):
        Ftr = Fv.clone().requires_grad_(need_grad)
        nck, K = len(cf["ckpt"]), cf["K"]
        acc = [0.0] * nck
        hard_acc, lr_pick = [0.0] * nck, []
        tot, cnt = 0.0, 0
        G = torch.zeros_like(Fv) if need_grad else None
        kappa = a.cf_curv
        single = list(cf["sched"].keys())[0] if len(cf["sched"]) == 1 else "grid"
        for k in range(a.views):
            for fold in range(K):
                ia = torch.nonzero(cf["fold"] != fold).squeeze(1)
                ib = torch.nonzero(cf["fold"] == fold).squeeze(1)
                yb, Hb = Y[ib], cf["H1"][ib]
                Ya = F.one_hot(Y[ia], C).float() - 1.0 / C
                ones = torch.ones(len(ia), 1, device=dev)
                for W0 in cf["W0s"]:
                    if single != "grid":
                        eta = single
                    else:
                        with torch.no_grad():
                            A_ = torch.cat([Ftr[k, ia].detach(), ones], 1)
                            M_, S_ = A_.t() @ A_ / len(ia), Ya.t() @ A_ / len(ia)
                            lam_max = torch.linalg.eigvalsh(M_.double())[-1].item()
                            best = (None, -1.0)
                            for eta_, lrs in cf["sched"].items():
                                if eta_ * kappa * lam_max / C > 2.0:
                                    continue
                                Ws = cf_run(M_, S_, lrs, cf["ckpt"], a.mom, a.wd, kappa, W0, False)
                                val = sum(cf_score(Wt, Hb, yb)[0].item() for Wt in Ws) / nck
                                if val > best[1]:
                                    best = (eta_, val)
                            assert best[1] >= 0, "cfprobe: every lr of the grid diverges"
                        eta = best[0]
                    with torch.enable_grad() if need_grad else torch.no_grad():
                        A_ = torch.cat([Ftr[k, ia], ones], 1)
                        M_, S_ = A_.t() @ A_ / len(ia), Ya.t() @ A_ / len(ia)
                        Ws = cf_run(M_, S_, cf["sched"][eta], cf["ckpt"], a.mom, a.wd, kappa, W0, need_grad)
                        Jkf = 0.0
                        for t, Wt in enumerate(Ws):
                            sc, h_ = cf_score(Wt, Hb, yb)
                            Jkf = Jkf - sc / nck
                            acc[t] += sc.item()
                            hard_acc[t] += h_
                    if need_grad:
                        g, = torch.autograd.grad(Jkf, Ftr)
                        G += g
                    tot += Jkf.item()
                    cnt += 1
                    lr_pick.append(eta)
        J = tot / cnt
        lastobj.clear()
        lastobj["cf_J"] = J
        lastobj["cf_soft"] = [x / cnt for x in acc]
        lastobj["cf_hard"] = [100.0 * x / cnt for x in hard_acc]
        lastobj["cf_lr"] = lr_pick
        if need_grad:
            G = G / cnt
            lastobj["gn_cf"] = G.abs().sum().item()
        Jtot = J
        if a.logpr_beta != 0:
            assert plan is not None
            Ft2 = Fv.clone().requires_grad_(need_grad)
            with torch.enable_grad() if need_grad else torch.no_grad():
                lp = 0.0
                for k, bi, _ in plan:
                    lp = lp + log_pr_group(Ft2[k, bi])
                lp = lp / len(plan)
            lastobj["logpr"] = lp.item()
            if need_grad:
                Gl, = torch.autograd.grad(lp, Ft2)
                lastobj["gn_w"], lastobj["gn_lp"] = G.abs().sum().item(), Gl.abs().sum().item()
                if a.lp_equal_vote:
                    lastobj["gn_lp_raw"] = lastobj["gn_lp"]
                    Gl = Gl * (G.abs().sum() / Gl.abs().sum().clamp_min(1e-30))
                    lastobj["gn_lp"] = Gl.abs().sum().item()
                G = G - a.logpr_beta * Gl
            Jtot = J - a.logpr_beta * lp.item()
        return Jtot, lastobj["cf_hard"], G

    def objective(Fv, need_grad, plan=None):
        if a.objective == "cfprobe":
            return cf_objective(Fv, need_grad, plan)
        fold = torch.randperm(N, device=dev, generator=gen) < N // 2
        G = torch.zeros_like(Fv) if need_grad else None
        Js, accs = [], []
        for f in (True, False):
            tr = torch.nonzero(fold == f).squeeze(1)
            ho = torch.nonzero(fold != f).squeeze(1)
            Ftr = Fv[:, tr].clone().requires_grad_(need_grad)
            J, acc = simulate(Ftr, Y[tr], H[ho], Y[ho], a, gen, need_grad)
            if need_grad:
                g, = torch.autograd.grad(J, Ftr)
                G[:, tr] = g
            Js.append(J.item())
            accs.append(acc)
        return sum(Js) / 2, [sum(x) / 2 for x in zip(*accs)], G

    meta = {k: (v if not isinstance(v, list) else list(v)) for k, v in vars(a).items()}
    meta["N"] = N

    if a.mode == "sim":
        t0 = time.time()
        Fv, plan_s = extract(None)
        J, accs, _ = objective(Fv, False, plan_s)
        tag = "theta_I" if a.ckpt else "theta_0"
        print(f"[sim] {tag}  J {J:.4f}  held-out acc @ep10..50 "
              + " ".join(f"{x:.2f}" for x in accs) + f"  mean {sum(accs) / len(accs):.2f}"
              + f"  ({time.time() - t0:.0f}s)", flush=True)
        return

    eps = a.eps / 255.0
    delta = torch.zeros(X.shape, dtype=torch.float32, device=dev)
    gbuf = torch.zeros_like(delta)
    hist = []
    for step in range(a.steps):
        t0 = time.time()
        Fv, plan = extract(delta)
        J, accs, G = objective(Fv, True, plan)
        gbuf.zero_()
        for k, bi, v in plan:
            d = delta[bi].clone().requires_grad_(True)
            f = feat(pe.apply_view((X[bi].float() / 255 + d).clamp(0, 1), v))
            gd, = torch.autograd.grad(f, d, grad_outputs=G[k, bi])
            gbuf[bi] += gd
        with torch.no_grad():
            for c0 in range(0, N, 1024):
                sl = slice(c0, min(c0 + 1024, N))
                x01 = X[sl].float() / 255
                if a.sparse_rho > 0:
                    d = -eps * gbuf[sl].sign() * pe.group_topk_mask(gbuf[sl], a.sparse_rho)
                else:
                    d = (delta[sl] - a.alpha * eps * gbuf[sl].sign()).clamp(-eps, eps)
                delta[sl] = (x01 + d).clamp(0, 1) - x01
                del x01, d
        rec = {"J": J, "acc": accs}
        ltxt = ""
        if a.logpr_beta != 0:
            rec["cf_J_term"], rec["logpr"] = lastobj["cf_J"], lastobj["logpr"]
            rec["gn_w"], rec["gn_lp"] = lastobj["gn_w"], lastobj["gn_lp"]
            if "gn_lp_raw" in lastobj:
                rec["gn_lp_raw"] = lastobj["gn_lp_raw"]
            ltxt = (f"  Jcf {lastobj['cf_J']:.4f} lp {lastobj['logpr']:.3f}"
                    f" r {a.logpr_beta * lastobj['gn_lp'] / max(lastobj['gn_w'], 1e-12):.2f}")
        if a.objective == "cfprobe":
            rec["cf_J"], rec["cf_soft"], rec["cf_hard"], rec["cf_lr"] = (lastobj["cf_J"], lastobj["cf_soft"],
                                                                         lastobj["cf_hard"], lastobj["cf_lr"])
            rec["gn_cf"] = lastobj.get("gn_cf")
            ltxt = (f"  cfJ {lastobj['cf_J']:.4f} soft " + " ".join(f"{x:.3f}" for x in lastobj["cf_soft"])
                    + (f"  lr* {sorted(set(lastobj['cf_lr']))}" if a.cn_view else "") + ltxt)
        hist.append(rec)
        print(f"  step {step + 1}/{a.steps}  J {J:.4f}  sim held-out acc "
              + " ".join(f"{x:.2f}" for x in accs) + ltxt
              + f"  ({time.time() - t0:.0f}s)", flush=True)

    D8 = torch.empty_like(X, dtype=torch.int8)
    for c0 in range(0, N, 1024):
        sl = slice(c0, min(c0 + 1024, N))
        D8[sl] = (torch.round((X[sl].float() / 255 + delta[sl]).clamp(0, 1) * 255) - X[sl].float()).to(torch.int8)
    del gbuf
    Fv, plan_f = extract(D8.float() / 255)
    J, accs, _ = objective(Fv, False, plan_f)
    meta["J_hist"] = hist
    meta["J_final_uint8"], meta["acc_final_uint8"] = J, accs
    if a.logpr_beta != 0:
        meta["cf_J_final_uint8"] = lastobj["cf_J"]
        meta["logpr_final_uint8"] = lastobj["logpr"]
    if a.objective == "cfprobe":
        meta["cf_final_uint8"] = {k: lastobj[k] for k in ("cf_J", "cf_soft", "cf_hard", "cf_lr")}
        meta.update(cf_meta)
        fe_dg = copy.deepcopy(fe).eval()

        def _ev(dl):
            with torch.no_grad():
                out = []
                for k in range(0, N, 256):
                    x = X[k:k + 256].float() / 255
                    if dl is not None:
                        x = (x + dl[k:k + 256]).clamp(0, 1)
                    out.append(fe_dg((pe.center_view(x) - mt) / st))
                return torch.cat(out)
        meta["evalview_clean"] = feat_stats(_ev(None), Y, C)
        meta["evalview_final_uint8"] = feat_stats(_ev(D8.float() / 255), Y, C)
        del fe_dg
        print("[evalview] clean " + json.dumps(meta["evalview_clean"]), flush=True)
        print("[evalview] final " + json.dumps(meta["evalview_final_uint8"]), flush=True)
    meta["peak_mem_gb"] = torch.cuda.max_memory_allocated() / 2 ** 30
    meta["bn_running_stat_drift"] = max((fe.state_dict()[k] - v).abs().max().item()
                                        for k, v in run_state.items())
    save_set(a, ds, X, D8, meta)


if __name__ == "__main__":
    main()
