import argparse
import csv
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import prompts as P


def build(key, dst, prompt, src_root, force):
    src = os.path.join(src_root, "data", key)
    if not os.path.isdir(src):
        raise SystemExit(f"missing source image dir {src}")
    pngs = sorted((f for f in os.listdir(src) if f.endswith(".png")),
                  key=lambda f: int(os.path.splitext(f)[0]))
    if len(pngs) != 100:
        raise SystemExit(f"{src}: expected 100 png, found {len(pngs)}")

    if os.path.isdir(dst):
        if not force:
            print(f"  exists, skip: {dst}")
            return False
        shutil.rmtree(dst)
    os.makedirs(dst)

    for f in pngs:
        shutil.copy2(os.path.join(src, f), os.path.join(dst, f))
    with open(os.path.join(dst, "metadata.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["file_name", "prompt"])
        for f in pngs:
            w.writerow([f, prompt])
    print(f"  built {dst}  ({len(pngs)} images)")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", default=os.environ.get("OUT_ROOT", "attack_data"))
    ap.add_argument("--force", action="store_true",
                    help="rebuild dirs that already exist")
    a = ap.parse_args()

    imma_root = os.environ.get("IMMA_ROOT")
    if not imma_root:
        raise SystemExit("IMMA_ROOT unset -- point it at the IMMA data root")

    built = 0
    for channel, key in P.BUILDABLE:
        template, prompt = P.SPEC[channel][key]
        dst = P.resolve(template, imma_root, a.out_root)
        print(f"\n=== {P.ARTIST[key]}  [{channel}]")
        print(f"  prompt: {prompt}")
        built += build(key, dst, prompt, imma_root, a.force)
    print(f"\nbuilt {built} dataset(s); "
          f"{len(P.BUILDABLE) - built} already present")


if __name__ == "__main__":
    main()
