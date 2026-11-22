import os
import sys

import torch

IMMA_ROOT = os.environ["IMMA_ROOT"]
ESD = {"vangogh": "VanGogh", "tyleredlin": "TylerEdlin", "kilianeng": "KilianEng",
       "thomaskinkade": "ThomasKinkade", "kellymckernan": "KellyMcKernan"}
PROJ = ("to_q.weight", "to_k.weight", "to_v.weight", "to_out.0.weight", "to_out.0.bias")


def attention_state(key, ck):
    esd = torch.load(f"{IMMA_ROOT}/diffusers-{ESD[key]}-ESDx1-UNET.pt", map_location="cpu")
    sd = {k: v for k, v in esd.items() if k.endswith(PROJ) and (".attn1." in k or ".attn2." in k)}
    del esd
    if ck is not None:
        d = torch.load(ck, map_location="cpu")
        assert set(d) == set(sd), f"{ck}: keys differ from the ESD attention tensors"
        sd.update(d)
    assert len(sd) == 160 and all(sum(k.endswith(p) for k in sd) == 32 for p in PROJ), "expected 32 x 5 tensors"
    return sd


def transform(sd, c):
    c = float(c)
    out = {}
    for k, w in sd.items():
        if k.endswith(("to_q.weight", "to_out.0.weight")):
            out[k] = w * c
        elif k.endswith(("to_k.weight", "to_v.weight")):
            out[k] = w / c
        else:
            out[k] = w.clone()
    return out


def main():
    key, ck, c, path = sys.argv[1:5]
    ck = None if ck == "none" else ck
    sd = transform(attention_state(key, ck), c)
    tmp = path + ".tmp"
    torch.save(sd, tmp)
    os.replace(tmp, path)
    print(f"[reparam] {key} {ck} c={c} -> {path} ({len(sd)} tensors)", flush=True)


if __name__ == "__main__":
    main()
