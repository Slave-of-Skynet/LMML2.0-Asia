"""China (oracle-bone rubbings): count the crate glyphs with a sliding-window classifier.

Why this design: the labelled boxes give 14,149 ready examples of "this window shows glyph X".  A small
network that classifies ONE window (one of the 8 crate glyphs / some other character / background) learns
from them in minutes on a CPU.  To find glyphs in a new rubbing the same network is slid over the image at
several window sizes; windows it calls "crate glyph" with high confidence are kept, overlapping ones merged.

    python window_clf.py train   --data ../task_china
    python window_clf.py val     --data ../task_china
    python window_clf.py count   --data ../task_china      # writes counts.csv
"""
import argparse, math, random, time, sys
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image

WIN, CTX = 48, 80                     # the network judges the central 48x48 of an 80x80 patch (it sees context)
SIZES = (52, 64, 78, 96, 118, 145, 178)   # window sides (original pixels) tried when scanning a rubbing
N_BG = 9                              # class ids: 0..7 crate glyphs, 8 other character, 9 background

def gray(path):
    return np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0

class Net(nn.Module):
    def __init__(self, n=10):
        super().__init__()
        def b(i, o, s=1): return nn.Sequential(nn.Conv2d(i, o, 3, s, 1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))
        self.f = nn.Sequential(b(1, 32, 2), b(32, 32), b(32, 64, 2), b(64, 64), b(64, 128, 2), b(128, 128), b(128, 128))   # stride 8
        self.h = nn.Sequential(nn.Conv2d(128, 256, 6, bias=False), nn.BatchNorm2d(256), nn.ReLU(inplace=True), nn.Dropout2d(0.2), nn.Conv2d(256, n, 1))
    def forward(self, x):             # x: B x 1 x H x W  ->  logits at stride 8 for every 48x48 window
        return self.h(self.f(x))

def crop_resize(img, cx, cy, side, out=CTX):
    """Square patch of `side*CTX/WIN` pixels centred at (cx, cy), resized so that `side` becomes WIN."""
    full = side * CTX / WIN; x0, y0 = cx - full / 2, cy - full / 2
    H, W = img.shape
    im = Image.fromarray((img * 255).astype(np.uint8))
    patch = im.crop((int(round(x0)), int(round(y0)), int(round(x0 + full)), int(round(y0 + full)))).resize((out, out), Image.BILINEAR)   # PIL pads outside with 0
    return np.asarray(patch, np.float32) / 255.0

def iou_center_ok(cx, cy, side, box):
    x, y, w, h = box; bx, by = x + w / 2, y + h / 2
    return abs(cx - bx) <= 0.18 * side and abs(cy - by) <= 0.18 * side and 0.95 * max(w, h) <= side <= 1.75 * max(w, h)

def load(data):
    labels = pd.read_csv(data / "train_labels.csv"); crates = pd.read_csv(data / "crates.csv", encoding="utf-8-sig")
    classes = {g: i for i, g in enumerate(crates.glyph_id)}
    names = sorted(labels.image.unique()); random.Random(1).shuffle(names)
    by = {n: g[["x", "y", "w", "h", "glyph_id"]].values.tolist() for n, g in labels.groupby("image")}
    return labels, crates, classes, names, by

def make_samples(imgs, by, classes, names, rng, n_other=9000, n_bg=12000, rep=4):
    X, Y = [], []
    target = [(n, b) for n in names for b in by[n] if b[4] in classes]
    other = [(n, b) for n in names for b in by[n] if b[4] not in classes]
    def pos(n, b):
        x, y, w, h, g = b; side = max(w, h) * rng.uniform(1.0, 1.6)
        return crop_resize(imgs[n], x + w / 2 + rng.uniform(-0.12, 0.12) * side, y + h / 2 + rng.uniform(-0.12, 0.12) * side, side)
    for n, b in target:
        for _ in range(rep): X.append(pos(n, b)); Y.append(classes[b[4]])
    for n, b in rng.sample(other, min(n_other, len(other))): X.append(pos(n, b)); Y.append(8)
    k = 0
    while k < n_bg:
        n = rng.choice(names); img = imgs[n]; H, W = img.shape; side = rng.choice(SIZES) * rng.uniform(0.85, 1.2)
        if rng.random() < 0.5 and by[n]:              # hard negatives: badly placed / badly sized windows near a character
            x, y, w, h, g = rng.choice(by[n]); cx = x + w / 2 + rng.uniform(-0.9, 0.9) * max(w, h); cy = y + h / 2 + rng.uniform(-0.9, 0.9) * max(w, h)
            if rng.random() < 0.3: side = max(w, h) * rng.choice([0.45, 0.6, 2.4, 3.0])
        else:
            cx, cy = rng.uniform(0, W), rng.uniform(0, H)
        if any(iou_center_ok(cx, cy, side, b[:4]) for b in by[n]): continue
        X.append(crop_resize(img, cx, cy, side)); Y.append(N_BG); k += 1
    X = np.stack(X)[:, None]; Y = np.array(Y)
    X = np.clip((X - 0.5) * np.array([rng.uniform(0.75, 1.3) for _ in range(len(X))], np.float32)[:, None, None, None] + 0.5, 0, 1)
    return torch.from_numpy(X.astype(np.float32)), torch.from_numpy(Y)

def scan(net, img, classes_n=8, thr=0.5):
    """Detections [(score, cls, cx, cy, side)] of crate glyphs in one rubbing (original pixel coordinates)."""
    dets = []
    for side in SIZES:
        f = WIN / side; H, W = img.shape
        im = Image.fromarray((img * 255).astype(np.uint8)).resize((max(8, round(W * f)), max(8, round(H * f))), Image.BILINEAR)
        a = np.asarray(im, np.float32) / 255.0; pad = (CTX - WIN) // 2 + WIN // 2
        x = torch.zeros(1, 1, a.shape[0] + 2 * pad, a.shape[1] + 2 * pad); x[0, 0, pad:pad + a.shape[0], pad:pad + a.shape[1]] = torch.from_numpy(a)
        p = torch.softmax(net(x)[0], 0)[:classes_n]                      # window k is centred at 8*k + 24 - (pad - 16) ... see below
        sc, cl = p.max(0)
        ys, xs = torch.nonzero(sc >= thr, as_tuple=True)
        for y, xx in zip(ys.tolist(), xs.tolist()):
            # output cell (y, xx) judges rows [8y, 8y+48) of the padded image, so its centre is 8*y + WIN/2
            cx = (8 * xx + WIN / 2 - pad) / f; cy = (8 * y + WIN / 2 - pad) / f
            dets.append((float(sc[y, xx]), int(cl[y, xx]), cx, cy, side))
    dets.sort(reverse=True); keep = []
    for d in dets:                                                       # merge overlapping windows: keep the most confident
        if all(math.hypot(d[2] - k[2], d[3] - k[3]) > 0.45 * min(d[4], k[4]) for k in keep): keep.append(d)
    return keep

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("mode"); ap.add_argument("--data", type=Path, default=Path("../task_china"))
    ap.add_argument("--epochs", type=int, default=10); ap.add_argument("--val", type=int, default=300); ap.add_argument("--thr", type=float, default=0.5); ap.add_argument("--nval", type=int, default=300); ap.add_argument("--ckpt", default="window_clf.pt")
    a = ap.parse_args(); torch.set_num_threads(8); torch.manual_seed(0)
    labels, crates, classes, names, by = load(a.data); val_names, train_names = names[:a.val], names[a.val:]
    if a.mode == "train":
        t0 = time.time(); rng = random.Random(0)
        imgs = {n: gray(a.data / "train" / "images" / n) for n in train_names}
        net = Net(); opt = torch.optim.AdamW(net.parameters(), 2e-3, weight_decay=5e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
        w = torch.ones(10); w[N_BG] = 1.0
        for ep in range(a.epochs):
            X, Y = make_samples(imgs, by, classes, train_names, rng); perm = torch.randperm(len(X)); net.train(); tot = acc = 0
            for i in range(0, len(X) - 63, 64):
                idx = perm[i:i + 64]; out = net(X[idx])[:, :, 2, 2]              # centre window of the 80x80 patch
                loss = F.cross_entropy(out, Y[idx], weight=w, label_smoothing=0.05); opt.zero_grad(); loss.backward(); opt.step()
                tot += loss.item(); acc += (out.argmax(1) == Y[idx]).float().mean().item()
            sched.step(); k = len(X) // 64
            print(f"epoch {ep+1}/{a.epochs} samples {len(X)} loss {tot/k:.3f} acc {acc/k:.3f} time {(time.time()-t0)/60:.1f} min", flush=True)
            torch.save({"state": net.state_dict(), "classes": classes}, "window_clf.pt")
        return
    ck = torch.load(a.ckpt); net = Net(); net.load_state_dict(ck["state"]); net.eval()
    inv = {v: k for k, v in classes.items()}
    if a.mode == "val":
        res = []
        with torch.no_grad():
            for n in val_names[:a.nval]:
                res.append((scan(net, gray(a.data / "train" / "images" / n), thr=0.3), by[n]))
        for thr in (0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
            print(f"--- threshold {thr}")
            for g, k in classes.items():
                tp = fp = fn = 0; pc = []; gc = []
                for dets, boxes in res:
                    gt = [b for b in boxes if b[4] == g]; used = set(); d = [x for x in dets if x[1] == k and x[0] >= thr]
                    for s, c, cx, cy, side in d:
                        hit = [i for i, (x, y, w, h, _) in enumerate(gt) if i not in used and abs(cx - x - w / 2) < 0.5 * max(w, h) and abs(cy - y - h / 2) < 0.5 * max(w, h)]
                        if hit: used.add(hit[0]); tp += 1
                        else: fp += 1
                    fn += len(gt) - len(used); pc.append(len(d)); gc.append(len(gt))
                err = [sum(pc[i:i + 30]) - sum(gc[i:i + 30]) for i in range(0, len(pc), 30)]
                print(f"   {g}: P {tp/max(1,tp+fp):.2f} R {tp/max(1,tp+fn):.2f} | per-set count error (pred-true) {err}")
        return
    if a.mode == "count":
        rows = []
        with torch.no_grad():
            for crate, g in zip(crates.crate, crates.glyph_id):
                k = classes[g]
                for sd in sorted((a.data / "crates" / crate).iterdir()):
                    per_img = [sum(1 for d in scan(net, gray(f), thr=0.3) if d[1] == k and d[0] >= a.thr) for f in sorted(sd.glob("*.jpg"))]
                    rows.append([crate, sd.name, g, sum(per_img)]); print(crate, sd.name, g, sum(per_img), flush=True)
        pd.DataFrame(rows, columns=["crate", "set", "glyph_id", "count"]).to_csv("counts.csv", index=False)

if __name__ == "__main__":
    main()
