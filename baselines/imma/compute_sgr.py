import os, glob, csv, argparse
import pandas as pd, torch
from PIL import Image
from torchvision import transforms

IMSIZE = 64
_loader = transforms.Compose([transforms.Resize((IMSIZE, IMSIZE), antialias=True),
                              transforms.ToTensor()])


def _load(path):
    img = Image.open(path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return _loader(img)


def _epochs(gen_dir):
    files = glob.glob(os.path.join(gen_dir, "*.png"))
    return sorted({os.path.basename(f).split("_")[0] for f in files
                   if "_" in os.path.basename(f) and os.path.basename(f).split("_")[0].isdigit()}, key=int)


def compute_lpips_csv(ref_dir, base_dir, imma_dir, out_csv, device, model):
    refs = torch.stack([_load(f) for f in sorted(sum([glob.glob(os.path.join(ref_dir, e)) for e in ("*.png","*.jpg","*.jpeg")], []))]).to(device)
    R = refs.shape[0]

    def per_epoch(gen_dir):
        out = {}
        with torch.no_grad():
            for ep in _epochs(gen_dir):
                gens = sorted(glob.glob(os.path.join(gen_dir, f"{ep}_*.png")))
                if not gens:
                    continue
                tot, cnt = 0.0, 0
                for g in gens:
                    gt = _load(g).unsqueeze(0).to(device).repeat(R, 1, 1, 1)
                    tot += model(gt, refs).sum().item()
                    cnt += R
                out[ep] = 1.0 - tot / cnt
        return out

    wo, wi = per_epoch(base_dir), per_epoch(imma_dir)
    eps = sorted(set(wo) & set(wi), key=int)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "w/o IMMA", "w/ IMMA"])
        for e in eps:
            w.writerow([e, wo[e], wi[e]])


def sgr_stats(csv_path):
    df = pd.read_csv(csv_path)
    ep_col = df.columns[0]
    df["_e"] = df[ep_col].astype(int)
    df = df.sort_values("_e").reset_index(drop=True)
    sgr = (df["w/o IMMA"] - df["w/ IMMA"]) / df["w/o IMMA"] * 100.0
    return {"mean": float(sgr.mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True,
                    help="dir with relearn/{without_imma,with_imma} and eval/{clip,dino}.csv")
    ap.add_argument("--reference_dir", required=True, help="dir of reference (clean) images")
    args = ap.parse_args()

    evald = os.path.join(args.run_dir, "eval")
    base = os.path.join(args.run_dir, "relearn", "without_imma")
    imma = os.path.join(args.run_dir, "relearn", "with_imma")

    import lpips
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = lpips.LPIPS(net="alex").to(device).eval()
    compute_lpips_csv(args.reference_dir, base, imma, os.path.join(evald, "lpips.csv"), device, model)

    print(f"\nSGR  ({args.run_dir}):")
    print(f"  {'metric':6s} {'mean':>13s}")
    with open(os.path.join(evald, "sgr_summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "mean_SGR"])
        for m in ["lpips", "clip", "dino"]:
            s = sgr_stats(os.path.join(evald, m + ".csv"))
            print(f"  {m.upper():6s} {s['mean']:>+12.2f}%")
            w.writerow([m.upper(), f"{s['mean']:.2f}"])
    print(f"saved -> {os.path.join(evald, 'sgr_summary.csv')}")


if __name__ == "__main__":
    main()
