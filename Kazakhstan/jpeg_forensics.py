"""JPEG-history forensics: pixels that never went through a stamp's first JPEG.

Official hint (2026-11-15): every stamp was scanned and JPEG-compressed
*before* it was modified, and the page was saved once more afterwards at high
quality.  Most modified pixels never went through the first compression of
their stamp.

Consequence used here.  An untouched 8x8 block of a stamp, taken on the
stamp's own first-compression grid, has DCT coefficients that sit (up to the
small noise of the final quality-97 page save) on the lattice ``k * q_uv`` of
the first quantisation table.  A block containing edited pixels does not.
Nothing else is needed: no original stamp, no catalogue, no album label.

Steps per stamp crop (all automatic):

1. ``estimate_grid`` - sliding 8x8 DCT (``cv2.filter2D`` with the 64 DCT
   basis kernels) gives the coefficients at every grid phase.  For every
   phase (dy, dx) in 0..7 the mean ``cos(2*pi*c/q)`` over the 15
   lowest-frequency luminance coefficients is computed for coarse qualities
   (70..95); the phase with the largest contrast against the median phase
   wins, then every IJG quality Q1 in 50..98 is scanned on that phase.  Stamps are pasted onto the page without resampling, so the
   phase differs from stamp to stamp (measured: Q1 82-92 on the exhibits).
2. ``consistency_blocks`` (training-free) - per block on that grid, the
   weighted mean of ``cos(2*pi*c/q)`` over the ``NK`` lowest zig-zag
   coefficients; the weight of a coefficient is its median cosine over the
   stamp (its reliability on mostly-untouched content).  Inconsistency =
   1 - score; box-smoothed with ``WIN`` px.
3. ``block_features`` (learned variant) - the same per-coefficient lattice
   phases plus |c|/q magnitudes for Y (21 coeffs) and Cb/Cr (6 each, first
   compression measured to be 4:4:4), block texture and stamp-relative
   phases.  ``HistGradientBoostingClassifier`` trained on exhibit blocks with
   grouped (whole-image) folds; maps of an exhibit always come from a model
   that never saw it.  Album maps use the model fitted on all 30 exhibits.

Parameters ``NK = 28``, weighted, ``WIN = 17`` were chosen by nested
5-fold selection on the exhibits (chosen in 3 of 5 outer folds; nested
held-out top-1 21/30).

Usage::

    python jpeg_forensics.py exhibits --data DATA --output outputs/jpeg
    python jpeg_forensics.py albums   --data DATA --output outputs/jpeg
"""
import argparse
import json
import pickle
from pathlib import Path

import cv2
import numpy as np

from geometry import exhibit_box, stamps

LUM = np.array([16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55,
                14, 13, 16, 24, 40, 57, 69, 56, 14, 17, 22, 29, 51, 87, 80, 62,
                18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113, 92,
                49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99],
               float).reshape(8, 8)
CHR = np.full((8, 8), 99.0)
CHR[:4, :4] = np.array([17, 18, 24, 47, 18, 21, 26, 66, 24, 26, 56, 99, 47, 66, 99, 99]).reshape(4, 4)
DCT = np.array([[np.sqrt((1 if k == 0 else 2) / 8) * np.cos((2 * n + 1) * k * np.pi / 16)
                 for n in range(8)] for k in range(8)])
ZZ = sorted([(u, v) for u in range(8) for v in range(8)],
            key=lambda t: (t[0] + t[1], t[0] if (t[0] + t[1]) % 2 else t[1]))
KERN = {(u, v): np.outer(DCT[u], DCT[v]).astype(np.float32) for u in range(8) for v in range(8)}
NK, WIN, MARGIN = 28, 17, 16
Q_RANGE = range(50, 99)
COARSE_Q = (70, 75, 80, 85, 90, 95)
NY, NC = 21, 6


def table(q, base=LUM):
    """IJG quantisation table for quality ``q``."""
    s = 5000 / q if q < 50 else 200 - 2 * q
    return np.clip(np.floor((base * s + 50) / 100), 1, 255)


def ycc(img):
    """JFIF YCbCr (level-shifted Y) of a BGR uint8 image, float64."""
    b, g, r = [img[..., i].astype(np.float64) for i in range(3)]
    y = 0.299 * r + 0.587 * g + 0.114 * b - 128
    cb = -0.168736 * r - 0.331264 * g + 0.5 * b
    cr = 0.5 * r - 0.418688 * g - 0.081312 * b
    return y, cb, cr


def coef_maps(plane, ks):
    """DCT coefficient of the 8x8 block whose top-left corner is each pixel."""
    p = plane.astype(np.float32)
    return {k: cv2.filter2D(p, -1, KERN[k], anchor=(0, 0), borderType=cv2.BORDER_REPLICATE)
            for k in ks}


def grid_blocks(a, dy, dx):
    h, w = a.shape[:2]
    return a[dy:dy + 8 * ((h - dy) // 8):8, dx:dx + 8 * ((w - dx) // 8):8]


def upsample(b, shape, dy, dx):
    out = np.full(shape, np.nan, np.float32)
    nby, nbx = b.shape
    out[dy:dy + 8 * nby, dx:dx + 8 * nbx] = np.repeat(np.repeat(b, 8, 0), 8, 1)
    return out


def _phase_means(f, q, ks, h, w):
    """Mean cos(2*pi*c/q) of every grid phase (8x8 array), complete blocks only."""
    t = table(q)
    hb, wb = (h // 8) * 8, (w // 8) * 8
    acc = np.zeros((hb, wb), np.float32)
    for k in ks:
        acc += np.cos(2 * np.pi * f[k][:hb, :wb] / t[k])
    return acc.reshape(hb // 8, 8, wb // 8, 8).mean(axis=(0, 2)) / len(ks)


def estimate_grid(y, qs=Q_RANGE, ks=ZZ[:15], coarse=COARSE_Q):
    """First-compression grid phase and IJG quality of one stamp.

    Stage 1 finds the phase with the largest contrast (best phase minus the
    median phase) over a coarse set of qualities using all 64 phases.
    Stage 2 scans every quality in ``qs`` on that phase only, measuring the
    contrast against four fixed reference phases (offsets (4,4), (4,0),
    (0,4), (2,6)); this is 1/13 of the full scan's cost.
    """
    f = coef_maps(y, ks)
    h, w = y.shape[0] - 7, y.shape[1] - 7
    best = None
    for q in coarse:
        ph = _phase_means(f, q, ks, h, w)
        dy, dx = np.unravel_index(ph.argmax(), ph.shape)
        c = float(ph[dy, dx] - np.median(ph))
        if best is None or c > best[0]:
            best = (c, int(dy), int(dx))
    _, dy, dx = best
    phases = [(dy, dx)] + [((dy + a) % 8, (dx + b) % 8) for a, b in ((4, 4), (4, 0), (0, 4), (2, 6))]
    blocks = {p: {k: grid_blocks(f[k][:h, :w], *p) for k in ks} for p in phases}
    out = None
    for q in qs:
        t = table(q)
        sc = [float(np.mean([np.cos(2 * np.pi * blocks[p][k] / t[k]).mean() for k in ks]))
              for p in phases]
        contrast = sc[0] - float(np.median(sc[1:]))
        if out is None or contrast > out["contrast"]:
            out = {"contrast": contrast, "score": sc[0], "dy": dy, "dx": dx, "Q": int(q)}
    return out


def consistency_blocks(y, est, nk=NK, weighted=True):
    """Training-free per-block lattice inconsistency on the estimated grid."""
    ks = ZZ[:nk]
    t = table(est["Q"])
    f = coef_maps(y, ks)
    cs = np.stack([np.cos(2 * np.pi * grid_blocks(f[k], est["dy"], est["dx"]) / t[k])
                   for k in ks], -1)
    w = np.clip(np.median(cs.reshape(-1, nk), 0), 0, 1) if weighted else np.ones(nk)
    return 1 - (cs * w).sum(-1) / (w.sum() + 1e-6)


def free_map(img, est=None, nk=NK, win=WIN):
    """Full-resolution smoothed training-free inconsistency map of a crop."""
    y, _, _ = ycc(img)
    est = est or estimate_grid(y)
    b = consistency_blocks(y, est, nk)
    m = upsample(b, y.shape, est["dy"], est["dx"])
    m = np.nan_to_num(m, nan=float(np.nanmedian(m)))
    return cv2.blur(m, (win, win)), est


def block_features(img, est=None):
    """Per-block feature tensor (nby, nbx, F) for the learned variant."""
    y, cb, cr = ycc(img)
    est = est or estimate_grid(y)
    dy, dx, q = est["dy"], est["dx"], est["Q"]
    feats = []
    for plane, base, n in ((y, LUM, NY), (cb, CHR, NC), (cr, CHR, NC)):
        t = table(q, base)
        ks = ZZ[:n]
        f = coef_maps(plane, ks)
        for k in ks:
            c = grid_blocks(f[k], dy, dx) / t[k]
            feats.append(np.cos(2 * np.pi * c))
            feats.append(np.minimum(np.abs(c), 4))
    yf = y.astype(np.float32)
    mean = grid_blocks(cv2.blur(yf, (8, 8), anchor=(0, 0)), dy, dx)
    sq = grid_blocks(cv2.blur(yf * yf, (8, 8), anchor=(0, 0)), dy, dx)
    feats.append(np.log1p(np.sqrt(np.maximum(sq - mean * mean, 0))))
    feats.append(mean / 128.0)
    f = np.stack(feats, -1).astype(np.float32)
    ycos = f[..., 0:2 * NY:2]
    rel = ycos - np.median(ycos.reshape(-1, NY), 0)
    f = np.concatenate([f, rel, np.full(f.shape[:2] + (1,), est["contrast"], np.float32)], -1)
    return f, est


def block_labels(mask, est, shape):
    nby, nbx = shape
    dy, dx = est["dy"], est["dx"]
    return mask[dy:dy + 8 * nby, dx:dx + 8 * nbx].reshape(nby, 8, nbx, 8).mean((1, 3))


def make_classifier():
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                          class_weight="balanced", random_state=0)


def fit_blocks(items):
    x = np.concatenate([it["f"].reshape(-1, it["f"].shape[-1]) for it in items])
    lab = np.concatenate([it["lab"].ravel() for it in items])
    keep = (lab > 0.5) | (lab == 0)
    return make_classifier().fit(x[keep], (lab[keep] > 0.5).astype(int))


def model_map(clf, f, est, shape, win=WIN):
    p = clf.predict_proba(f.reshape(-1, f.shape[-1]))[:, 1].reshape(f.shape[:2])
    m = upsample(p.astype(np.float32), shape, est["dy"], est["dx"])
    m = np.nan_to_num(m, nan=float(np.nanmedian(m)))
    return cv2.blur(m, (win, win))


def nms_peaks(heat, valid, k=5, radius=44):
    work = np.where(valid, heat, -np.inf).astype(np.float32)
    out = []
    for _ in range(k):
        iy, ix = np.unravel_index(np.argmax(work), work.shape)
        if not np.isfinite(work[iy, ix]):
            break
        out.append((int(ix), int(iy), float(work[iy, ix])))
        work[max(0, iy - radius):iy + radius + 1,
             max(0, ix - radius * 4 // 3):ix + radius * 4 // 3 + 1] = -np.inf
    return out


def exhibit_metrics(heat, mask, k=5):
    """AP, top-1/3/5 hit (peak within 4 px of the mask), false-peak count."""
    from sklearn.metrics import average_precision_score
    valid = np.zeros(mask.shape, bool)
    valid[MARGIN:-MARGIN, MARGIN:-MARGIN] = True
    dil = cv2.dilate(mask.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    pk = nms_peaks(heat, valid, k)
    hit = [bool(dil[y, x]) for x, y, _ in pk]
    first = hit.index(True) + 1 if True in hit else 0
    return {"ap": float(average_precision_score(mask[valid].ravel(), heat[valid].ravel())),
            "top1": int(first == 1), "top3": int(0 < first <= 3), "top5": int(0 < first <= 5),
            "rank": first, "false_peaks_before_hit": (first - 1) if first else len(pk)}


def load_exhibits(data):
    out = []
    for f in sorted((data / "exhibits").glob("exhibit_??.jpg")):
        image = cv2.imread(str(f))
        x, y, w, h = exhibit_box(image)
        mask = cv2.imread(str(f.with_name(f.stem + "_mask.png")), 0)[y:y + h, x:x + w] > 127
        out.append({"stem": f.stem, "img": image[y:y + h, x:x + w], "mask": mask})
    return out


def run_exhibits(args):
    out = args.output
    (out / "exhibit_free").mkdir(parents=True, exist_ok=True)
    (out / "exhibit_oof").mkdir(parents=True, exist_ok=True)
    items = load_exhibits(args.data)
    for it in items:
        it["f"], it["est"] = block_features(it["img"])
        it["lab"] = block_labels(it["mask"], it["est"], it["f"].shape[:2])
        it["free"], _ = free_map(it["img"], it["est"])
        np.save(out / "exhibit_free" / f"{it['stem']}.npy", it["free"].astype(np.float16))
        print("features", it["stem"], it["est"], flush=True)
    folds = [[j for j in range(len(items)) if j % 5 == k] for k in range(5)]
    for test in folds:
        clf = fit_blocks([items[j] for j in range(len(items)) if j not in test])
        for j in test:
            it = items[j]
            it["oof"] = model_map(clf, it["f"], it["est"], it["mask"].shape)
            np.save(out / "exhibit_oof" / f"{it['stem']}.npy", it["oof"].astype(np.float16))
    report = {"protocol": "grouped 5-fold by whole exhibit (j % 5); free map parameters fixed "
                          "(NK=28 weighted, WIN=17) by nested selection",
              "per_exhibit": {}}
    for name in ("free", "oof"):
        rows = {it["stem"]: exhibit_metrics(it[name], it["mask"]) for it in items}
        report[name] = {m: int(sum(r[m] for r in rows.values())) for m in ("top1", "top3", "top5")}
        report[name]["mean_ap"] = float(np.mean([r["ap"] for r in rows.values()]))
        report[name]["false_peaks_before_hit"] = int(sum(r["false_peaks_before_hit"] for r in rows.values()))
        for s, r in rows.items():
            report["per_exhibit"].setdefault(s, {"grid": items[[i["stem"] for i in items].index(s)]["est"]})[name] = r
        print(name, report[name], flush=True)
    clf = fit_blocks(items)
    with open(out / "block_model.pkl", "wb") as fh:
        pickle.dump(clf, fh)
    (out / "exhibit_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


def run_albums(args):
    out = args.output / "album_maps"
    out.mkdir(parents=True, exist_ok=True)
    with open(args.output / "block_model.pkl", "rb") as fh:
        clf = pickle.load(fh)          # written by the exhibits step of this script
    grids = {}
    for path in sorted((args.data / "albums").glob(f"{args.album or '*'}/*.jpg")):
        key = f"{path.parent.name}_{path.stem}"
        if args.reuse and (out / f"{key}_free.npy").exists():
            continue
        image = cv2.imread(str(path))
        free = np.full(image.shape[:2], np.nan, np.float32)
        model = np.full(image.shape[:2], np.nan, np.float32)
        grids[key] = []
        for bx, by, bw, bh in stamps(image):
            x, y, w, h = bx + 6, by + 6, bw - 12, bh - 12
            crop = image[y:y + h, x:x + w]
            f, est = block_features(crop)
            fm, _ = free_map(crop, est)
            free[y:y + h, x:x + w] = fm
            model[y:y + h, x:x + w] = model_map(clf, f, est, (h, w))
            grids[key].append({"box": [x, y, w, h], **est})
        np.save(out / f"{key}_free.npy", free.astype(np.float16))
        np.save(out / f"{key}_model.npy", model.astype(np.float16))
        print(key, [(g["Q"], round(g["contrast"], 3)) for g in grids[key]], flush=True)
    gpath = args.output / f"album_grids_{args.album or 'all'}.json"
    old = json.loads(gpath.read_text()) if gpath.exists() else {}
    old.update(grids)
    gpath.write_text(json.dumps(old, indent=1), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=["exhibits", "albums"])
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reuse", action="store_true")
    parser.add_argument("--album", help="albums mode: process one album directory only")
    args = parser.parse_args()
    run_exhibits(args) if args.mode == "exhibits" else run_albums(args)


if __name__ == "__main__":
    main()
