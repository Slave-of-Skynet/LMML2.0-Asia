"""Mongolia v2 sheep counter: CSRNet-style density regression on a pretrained VGG16-BN frontend.

Full-resolution (600x600) input, stride-8 density map (75x75), point-supervised targets built by
exact 8x8 mass binning + reflect-mode Gaussian (mass preserving). 5 image-level folds; fold 0 is
exactly the 56-image validation split of the original train_baseline.py (random.seed(42) shuffle).

  python train_v2.py --data <task_mongolia dir> --out <run dir> --fold 0 [--epochs 60]
Writes <out>/fold{k}.pt (best val MAE), <out>/fold{k}_val.csv (per-image predictions), log lines.
"""
import argparse
import os
import random
import time

import numpy as np
import pandas as pd
import scipy.ndimage
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from PIL import Image

STRIDE = 8
SCALE = 100.0
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def folds_for(images, k=5):
    rnd = random.Random(42)
    sh = sorted(images)
    random.seed(42)
    random.shuffle(sh)
    n = len(sh) // k
    return [sh[i * n:(i + 1) * n] if i < k - 1 else sh[i * n:] for i in range(k)]


class CSRNet(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        w = torchvision.models.VGG16_BN_Weights.IMAGENET1K_V1 if pretrained else None
        vgg = torchvision.models.vgg16_bn(weights=w)
        self.frontend = nn.Sequential(*list(vgg.features.children())[:33])
        cfg = [512, 512, 512, 256, 128, 64]
        layers, c = [], 512
        for o in cfg:
            layers += [nn.Conv2d(c, o, 3, padding=2, dilation=2), nn.BatchNorm2d(o), nn.ReLU(inplace=True)]
            c = o
        self.backend = nn.Sequential(*layers)
        self.head = nn.Conv2d(c, 1, 1)

    def forward(self, x):
        return F.relu(self.head(self.backend(self.frontend(x))))


def density_target(pts, h, w, sigma=1.0):
    gh, gw = h // STRIDE, w // STRIDE
    d = np.zeros((gh, gw), np.float32)
    if len(pts):
        xs = np.clip((pts[:, 0] / STRIDE).astype(int), 0, gw - 1)
        ys = np.clip((pts[:, 1] / STRIDE).astype(int), 0, gh - 1)
        np.add.at(d, (ys, xs), 1.0)
        d = scipy.ndimage.gaussian_filter(d, sigma=sigma, mode="reflect")
    return d


def to_tensor(arr):
    x = (arr.astype(np.float32) / 255.0 - MEAN) / STD
    return torch.from_numpy(x.transpose(2, 0, 1).copy())


class TrainSet(torch.utils.data.Dataset):
    def __init__(self, imgs, pts, crop=384, reps=4):
        self.imgs, self.pts, self.crop, self.reps = imgs, pts, crop, reps

    def __len__(self):
        return len(self.imgs) * self.reps

    def __getitem__(self, i):
        i %= len(self.imgs)
        im, p = self.imgs[i], self.pts[i].copy()
        s = random.uniform(0.75, 1.3)
        H, W = im.shape[:2]
        nh, nw = int(round(H * s / STRIDE)) * STRIDE, int(round(W * s / STRIDE)) * STRIDE
        pil = Image.fromarray(im).resize((nw, nh), Image.BILINEAR)
        a = np.asarray(pil)
        if len(p):
            p = p * np.array([nw / W, nh / H])
        c = min(self.crop, nh, nw)
        y0 = random.randrange(0, nh - c + 1, STRIDE) if nh > c else 0
        x0 = random.randrange(0, nw - c + 1, STRIDE) if nw > c else 0
        a = a[y0:y0 + c, x0:x0 + c]
        if len(p):
            p = p - np.array([x0, y0])
            keep = (p[:, 0] >= 0) & (p[:, 0] < c) & (p[:, 1] >= 0) & (p[:, 1] < c)
            p = p[keep]
        if random.random() < 0.5:
            a = a[:, ::-1]
            if len(p):
                p[:, 0] = c - 1e-3 - p[:, 0]
        if random.random() < 0.5:
            a = a[::-1]
            if len(p):
                p[:, 1] = c - 1e-3 - p[:, 1]
        a = a.astype(np.float32)
        if random.random() < 0.8:
            a = a * random.uniform(0.8, 1.2) + random.uniform(-15, 15)
            g = a.mean(2, keepdims=True)
            a = g + (a - g) * random.uniform(0.8, 1.2)
        a = np.clip(a, 0, 255)
        d = density_target(p, c, c)
        return to_tensor(a), torch.from_numpy(d)[None] * SCALE, float(len(p))


def load(data, names, pts_df):
    imgs, pts = [], []
    g = {k: v[["x", "y"]].to_numpy(float) for k, v in pts_df.groupby("image")}
    for n in names:
        imgs.append(np.asarray(Image.open(os.path.join(data, n)).convert("RGB")))
        pts.append(g.get(n, np.zeros((0, 2))))
    return imgs, pts


@torch.no_grad()
def predict(model, imgs, device, tta=True):
    model.eval()
    out = []
    for a in imgs:
        x = to_tensor(a)[None].to(device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            y = model(x).float().sum().item()
            if tta:
                y = 0.5 * (y + model(torch.flip(x, [3])).float().sum().item())
        out.append(y / SCALE)
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--crop", type=int, default=384)
    ap.add_argument("--all", action="store_true", help="train on all 280 images (no validation)")
    args = ap.parse_args()
    set_seed(42 + args.fold)
    os.makedirs(args.out, exist_ok=True)
    counts = pd.read_csv(os.path.join(args.data, "train_counts.csv"))
    pts_df = pd.read_csv(os.path.join(args.data, "train_points.csv"))
    folds = folds_for(list(counts.image))
    val_names = [] if args.all else folds[args.fold]
    tr_names = [n for n in sorted(counts.image) if n not in set(val_names)]
    tdir = os.path.join(args.data, "train")
    tr_imgs, tr_pts = load(tdir, tr_names, pts_df)
    va_imgs, va_pts = load(tdir, val_names, pts_df)
    va_true = np.array([len(p) for p in va_pts], float)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CSRNet().to(device)
    ds = TrainSet(tr_imgs, tr_pts, crop=args.crop, reps=2)
    dl = torch.utils.data.DataLoader(ds, batch_size=args.bs, shuffle=True, num_workers=0, drop_last=True)
    opt = torch.optim.AdamW([{"params": model.frontend.parameters(), "lr": args.lr * 0.5},
                             {"params": list(model.backend.parameters()) + list(model.head.parameters()), "lr": args.lr}],
                            weight_decay=1e-4)
    steps = args.epochs * len(dl)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[args.lr * 0.5, args.lr], total_steps=steps, pct_start=0.05)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")
    tag = "all" if args.all else f"fold{args.fold}"
    best = (1e9, None)
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        model.train()
        tl = 0.0
        for x, d, c in dl:
            x, d, c = x.to(device), d.to(device), c.to(device)
            with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                y = model(x)
            y = y.float()
            loss = F.mse_loss(y, d, reduction="sum") / x.shape[0] / SCALE \
                + 0.1 * F.l1_loss(y.sum((1, 2, 3)) / SCALE, c)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            tl += loss.item()
        msg = f"{tag} ep{ep:03d} loss={tl / len(dl):.3f} t={time.time() - t0:.0f}s"
        if val_names and (ep % 2 == 0 or ep > args.epochs - 15):
            pr = predict(model, va_imgs, device, tta=False)
            e = pr - va_true
            mae = np.abs(e).mean()
            msg += f" valMAE={mae:.2f} RMSE={np.sqrt((e ** 2).mean()):.2f} bias={e.mean():.2f}"
            if mae < best[0]:
                best = (mae, ep)
                torch.save(model.state_dict(), os.path.join(args.out, f"{tag}.pt"))
        print(msg, flush=True)
    if not val_names:
        torch.save(model.state_dict(), os.path.join(args.out, f"{tag}.pt"))
        return
    torch.save(model.state_dict(), os.path.join(args.out, f"{tag}_last.pt"))
    model.load_state_dict(torch.load(os.path.join(args.out, f"{tag}.pt"), map_location=device, weights_only=True))
    pr = predict(model, va_imgs, device, tta=True)
    e = pr - va_true
    pd.DataFrame({"image": val_names, "true": va_true, "pred": pr}).to_csv(os.path.join(args.out, f"{tag}_val.csv"), index=False)
    print(f"{tag} BEST ep{best[1]} TTA valMAE={np.abs(e).mean():.2f} RMSE={np.sqrt((e ** 2).mean()):.2f} "
          f"bias={e.mean():.2f} medAE={np.median(np.abs(e)):.2f}", flush=True)


if __name__ == "__main__":
    main()
