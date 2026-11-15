"""Predict sheep counts for the 333 survey images with one or more v2 checkpoints (averaged, hflip TTA).

  python predict_v2.py --data <task_mongolia dir> --ckpt a.pt [b.pt ...] --out counts.csv [--raw raw.csv]
"""
import argparse
import hashlib
import os
import sys

import numpy as np
import pandas as pd
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_v2 import CSRNet, predict  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--ckpt", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--raw", default=None)
    ap.add_argument("--split", default="survey")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.split == "survey":
        names = list(pd.read_csv(os.path.join(args.data, "survey_plots.csv")).image)
    else:
        names = list(pd.read_csv(os.path.join(args.data, "train_counts.csv")).image)
    imgs = [np.asarray(Image.open(os.path.join(args.data, args.split, n)).convert("RGB")) for n in names]
    preds = {}
    for c in args.ckpt:
        print("ckpt", c, hashlib.sha256(open(c, "rb").read()).hexdigest(), flush=True)
        m = CSRNet(pretrained=False).to(device)
        m.load_state_dict(torch.load(c, map_location=device, weights_only=True))
        preds[os.path.basename(c)] = predict(m, imgs, device, tta=True)
    raw = pd.DataFrame({"image": names, **preds})
    raw["mean"] = raw[list(preds)].mean(1)
    if args.raw:
        raw.to_csv(args.raw, index=False)
    out = pd.DataFrame({"image": names, "count": np.clip(np.rint(raw["mean"]), 0, None).astype(int)})
    out.to_csv(args.out, index=False, lineterminator="\n")
    print(out["count"].describe().round(2).to_string())


if __name__ == "__main__":
    main()
