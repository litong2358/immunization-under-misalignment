import argparse
import csv
import hashlib
import json
import os
import time

import numpy as np
import torch
from diffusers import DiffusionPipeline
from PIL import Image

ARTIST = {
    "vangogh": "Van Gogh",
    "tyleredlin": "Tyler Edlin",
    "kilianeng": "Kilian Eng",
    "thomaskinkade": "Thomas Kinkade",
    "kellymckernan": "Kelly McKernan",
}


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        h.update(fh.read())
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="stabilityai/stable-diffusion-xl-base-1.0")
    ap.add_argument("--keys", nargs="+", default=list(ARTIST))
    ap.add_argument("--out-template", required=True)
    ap.add_argument("--num", type=int, default=20)
    ap.add_argument("--seed-base", type=int, default=1234)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--gen-res", type=int, default=1024, help="SDXL native render size")
    ap.add_argument("--save-res", type=int, default=512, help="size written to disk")
    a = ap.parse_args()

    pipe = DiffusionPipeline.from_pretrained(a.model, torch_dtype=torch.float32)
    pipe = pipe.to("cuda")
    pipe.set_progress_bar_config(disable=True)
    guidance = 7.5

    for key in a.keys:
        prompt = f"An artwork by {ARTIST[key]}"
        out = a.out_template.format(key=key)
        if os.path.isfile(os.path.join(out, "manifest.json")):
            print(f"### {out} already complete, skipping", flush=True)
            continue
        os.makedirs(out, exist_ok=True)
        rows, t0, blanks = [], time.time(), []
        for i in range(a.num):
            g = torch.Generator(device="cuda").manual_seed(a.seed_base + i)
            img = pipe(prompt, height=a.gen_res, width=a.gen_res,
                       num_inference_steps=a.steps, guidance_scale=guidance,
                       generator=g).images[0]
            if a.save_res != a.gen_res:
                img = img.resize((a.save_res, a.save_res), Image.LANCZOS)
            fn = f"{i}.png"
            p = os.path.join(out, fn)
            img.save(p)
            if np.asarray(img.convert("L")).max() == 0:
                blanks.append(fn)
            rows.append({"file_name": fn, "prompt": prompt, "seed": a.seed_base + i,
                         "md5": md5(p)})
        if blanks:
            raise SystemExit(f"REFUSING {key}: all-black frames {blanks}")
        with open(os.path.join(out, "metadata.csv"), "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["file_name", "prompt"])
            w.writerows([[r["file_name"], r["prompt"]] for r in rows])
        with open(os.path.join(out, "manifest.json"), "w") as fh:
            json.dump({"model": a.model, "refiner": None, "prompt": prompt, "num": a.num,
                       "steps": a.steps, "guidance_scale": guidance,
                       "gen_res": a.gen_res, "save_res": a.save_res, "resample": "LANCZOS",
                       "dtype": "float32", "seed_base": a.seed_base,
                       "scheduler": type(pipe.scheduler).__name__, "images": rows}, fh, indent=1)
        print(f"DONE {key}: {a.num} images, {time.time() - t0:.0f}s -> {out}", flush=True)


if __name__ == "__main__":
    main()
