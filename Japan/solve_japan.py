#!/usr/bin/env python3
"""LMML 2.0 - Japan / City of Subtitles: camera (PRNU) attribution of the anonymous photos.

Pipeline
  1. Noise residual per photo: wavelet (db8, 4 levels) local-Wiener denoiser on each RGB channel,
     residual averaged over channels; subtitle boxes (pure-black overlays) and clipped pixels masked;
     row/column means removed.
  2. Sensor orientation: the same sensor fingerprint appears either upright or rotated by 180 degrees
     (landscape shots taken with the phone turned either way). For every credited author an EM loop
     assigns each photo an orientation so that all residuals are accumulated in one sensor frame.
  3. Fingerprint  K = W( sum T(w*I) / sum T(I^2) )  (T = orientation, W = DFT-domain Wiener filter
     that removes periodic JPEG-grid components). Score = max over orientations of the normalised
     correlation corr(w, I*T(K)) on unmasked pixels.
  4. Validation on the credited photos (leave-one-out and leave-one-week-out).
  5. Semi-supervised refinement: anonymous photos with a clear margin are added (in their detected
     orientation) to the fingerprints; every photo is always scored against fingerprints that exclude it.

Usage:  python solve_japan.py --dataset ./dataset --output ./outputs/japan [--workers 3] [--lowo]
The dataset folder must contain credited/<author>/*.jpg and anonymous/*.jpg.
"""
import argparse, csv, glob, json, os, sys, time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pywt
from PIL import Image
from scipy import ndimage

H, W = 960, 1280
ORIENT = {"id": lambda a: a, "r180": lambda a: a[::-1, ::-1]}


# ----------------------------------------------------------------------------- features
def load_rgb(path):
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)


def wavelet_residual(x, sigma0=3.0, levels=4, wave="db8"):
    coeffs = pywt.wavedec2(x, wave, level=levels, mode="periodization")
    out = [coeffs[0]]
    s2 = sigma0 ** 2
    for det in coeffs[1:]:
        nd = []
        for d in det:
            var = None
            for win in (3, 5, 7, 9):
                v = np.maximum(ndimage.uniform_filter(d * d, win) - s2, 0)
                var = v if var is None else np.minimum(var, v)
            nd.append(d * var / (var + s2))
        out.append(tuple(nd))
    return x - pywt.waverec2(out, wave, mode="periodization")[: x.shape[0], : x.shape[1]]


def zero_mean(a):
    a = a - a.mean(axis=0, keepdims=True)
    return a - a.mean(axis=1, keepdims=True)


def overlay_mask(rgb):
    """True where the pixel can carry sensor noise (not subtitle box, not clipped)."""
    box = ndimage.binary_closing(rgb.max(-1) < 3, np.ones((25, 25)))
    box = ndimage.binary_dilation(box, np.ones((9, 9)))
    sat = ndimage.binary_dilation(rgb.max(-1) > 250, np.ones((3, 3)))
    return ~(box | sat)


def feature_job(args):
    path, out = args
    if os.path.exists(out):
        return out
    rgb = load_rgb(path)
    res = np.mean([wavelet_residual(rgb[..., c]) for c in range(3)], axis=0)
    m = overlay_mask(rgb)
    v = zero_mean(res * m) * m
    np.savez(out, v=v.astype(np.float16), m=np.packbits(m))
    return out


class Store:
    def __init__(self, items, cache):
        self.items, self.cache = items, cache

    def load(self, i):
        z = np.load(os.path.join(self.cache, f"f{i:03d}.npz"))
        v = z["v"].astype(np.float32)
        m = np.unpackbits(z["m"])[: H * W].reshape(H, W).astype(np.float32)
        I = load_rgb(self.items[i]["path"]).mean(-1)
        return v, I, m


# ----------------------------------------------------------------------------- fingerprints
def wiener_dft(k):
    F = np.fft.fft2(k)
    P = np.abs(F)
    s = np.median(P)
    return np.real(np.fft.ifft2(F * s / np.maximum(P, s)))


def finalize(n, d):
    return wiener_dft(zero_mean(n / (d + 1.0)))


def contrib(store, i, o):
    v, I, m = store.load(i)
    f = ORIENT[o]
    return f(v * I), f(I * I * m)


def corr_masked(v, p, m):
    s = m > 0
    a, b = v[s], p[s]
    a = a - a.mean(); b = b - b.mean()
    return float(a @ b / np.sqrt((a @ a) * (b @ b) + 1e-30))


def score_all_orient(v, I, m, K):
    s = {o: corr_masked(v, I * ORIENT[o](K), m) for o in ORIENT}
    o = max(s, key=s.get)
    return s[o], o


def em_orientation(store, idx, iters=6):
    con = {i: {o: contrib(store, i, o) for o in ORIENT} for i in idx}
    ori = {i: "id" for i in idx}
    for _ in range(iters):
        n = sum(con[i][ori[i]][0] for i in idx); d = sum(con[i][ori[i]][1] for i in idx)
        new = {}
        for i in idx:
            K = finalize(n - con[i][ori[i]][0], d - con[i][ori[i]][1])
            new[i] = score_all_orient(*store.load(i), K)[1]
        if sum(o == "r180" for o in new.values()) > len(idx) / 2:  # majority defines the sensor frame
            new = {i: ("id" if o == "r180" else "r180") for i, o in new.items()}
        changed = sum(new[i] != ori[i] for i in idx)
        ori = new
        if changed == 0:
            break
    return ori


def build(store, members):
    """members: {i: (author, orientation)} -> per-author sums."""
    n, d = {}, {}
    for i, (a, o) in members.items():
        cn, cd = contrib(store, i, o)
        n[a] = n.get(a, 0) + cn; d[a] = d.get(a, 0) + cd
    return n, d


def score_photos(store, authors, n, d, members, photos):
    K = {a: finalize(n[a], d[a]) for a in authors}
    S = {}
    for i in photos:
        v, I, m = store.load(i)
        Ks = K
        if i in members:  # always exclude the photo itself from its own fingerprint
            a, o = members[i]
            cn, cd = contrib(store, i, o)
            Ks = dict(K); Ks[a] = finalize(n[a] - cn, d[a] - cd)
        S[i] = {a: score_all_orient(v, I, m, Ks[a]) for a in authors}
    return S, K


def best2(s):
    r = sorted(s.items(), key=lambda kv: -kv[1][0])
    return r[0], r[1]


def refine(store, authors, labelled, anon, S0, rounds=3, min_margin=0.0015, score_set=None):
    S = S0
    for r in range(rounds):
        pseudo = {}
        for i in anon:
            (a1, (s1, o1)), (_, (s2, _)) = best2(S[i])
            if s1 - s2 >= min_margin:
                pseudo[i] = (a1, o1)
        members = {**labelled, **pseudo}
        n, d = build(store, members)
        S_new, K = score_photos(store, authors, n, d, members, score_set if score_set is not None else list(S0))
        S = {**S, **S_new}
        print(f"  refinement round {r}: {len(pseudo)} anonymous photos used as pseudo-labels", flush=True)
    return S, K, members


# ----------------------------------------------------------------------------- evaluation
def confusion(S, items, idx, authors):
    cm = np.zeros((len(authors), len(authors)), int)
    for i in idx:
        t = authors.index(items[i]["author"]); p = authors.index(best2(S[i])[0][0])
        cm[t, p] += 1
    return cm


def metrics(cm, authors):
    out = {"accuracy": float(np.trace(cm) / cm.sum()), "per_author": {}}
    for k, a in enumerate(authors):
        tp = cm[k, k]; fp = cm[:, k].sum() - tp; fn = cm[k].sum() - tp
        out["per_author"][a] = {"precision": float(tp / max(tp + fp, 1)), "recall": float(tp / max(tp + fn, 1)), "support": int(cm[k].sum())}
    out["confusion"] = {"authors": authors, "matrix": cm.tolist()}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--lowo", action="store_true", help="also run leave-one-week-out validation (slow)")
    ap.add_argument("--rounds", type=int, default=3)
    args = ap.parse_args()
    t0 = time.time()
    os.makedirs(args.output, exist_ok=True)
    cache = os.path.join(args.output, "cache"); os.makedirs(cache, exist_ok=True)

    items = []
    for p in sorted(glob.glob(os.path.join(args.dataset, "credited", "*", "*.jpg"))):
        items.append({"path": p, "name": os.path.basename(p), "author": os.path.basename(os.path.dirname(p))})
    for p in sorted(glob.glob(os.path.join(args.dataset, "anonymous", "*.jpg"))):
        items.append({"path": p, "name": os.path.basename(p), "author": None})
    for it in items:
        it["week"] = int(it["name"][1:3])
    authors = sorted({it["author"] for it in items if it["author"]})
    cred = [i for i, it in enumerate(items) if it["author"]]
    anon = [i for i, it in enumerate(items) if not it["author"]]
    print(f"{len(cred)} credited photos ({', '.join(authors)}), {len(anon)} anonymous photos", flush=True)

    with ProcessPoolExecutor(args.workers) as ex:
        list(ex.map(feature_job, [(it["path"], os.path.join(cache, f"f{i:03d}.npz")) for i, it in enumerate(items)]))
    print(f"features ready ({time.time() - t0:.0f}s)", flush=True)
    store = Store(items, cache)

    # 1. sensor orientation inside each credited author
    orient = {}
    for a in authors:
        idx = [i for i in cred if items[i]["author"] == a]
        orient.update(em_orientation(store, idx))
        print(f"  orientation EM {a}: {sum(orient[i] == 'r180' for i in idx)}/{len(idx)} photos rotated 180 deg", flush=True)
    labelled = {i: (items[i]["author"], orient[i]) for i in cred}

    # 2. leave-one-out on credited photos (credited fingerprints only) + first anonymous scores
    n, d = build(store, labelled)
    S0, _ = score_photos(store, authors, n, d, labelled, cred + anon)
    m_loo = metrics(confusion(S0, items, cred, authors), authors)
    print(f"LOO (credited only, orientation-aware): accuracy {m_loo['accuracy']:.4f}", flush=True)

    # naive control: no orientation handling
    report = {"loo_credited_only": m_loo}

    # 3. semi-supervised refinement with anonymous photos
    S, K, members = refine(store, authors, labelled, anon, S0, rounds=args.rounds)
    m_ref = metrics(confusion(S, items, cred, authors), authors)
    report["loo_after_refinement"] = m_ref
    print(f"LOO after refinement: accuracy {m_ref['accuracy']:.4f}", flush=True)

    # 4. optional leave-one-week-out (whole credited week hidden from orientation EM, fingerprints and refinement)
    if args.lowo:
        held_pred = {}
        for wk in sorted({items[i]["week"] for i in cred}):
            train = [i for i in cred if items[i]["week"] != wk]; test = [i for i in cred if items[i]["week"] == wk]
            ori_w = {}
            for a in authors:
                ori_w.update(em_orientation(store, [i for i in train if items[i]["author"] == a]))
            lab_w = {i: (items[i]["author"], ori_w[i]) for i in train}
            nw, dw = build(store, lab_w)
            Sw0, _ = score_photos(store, authors, nw, dw, lab_w, anon + test)
            Sw, _, _ = refine(store, authors, lab_w, anon, Sw0, rounds=2, score_set=anon + test)
            for i in test:
                held_pred[i] = Sw[i]
            acc = np.mean([best2(Sw[i])[0][0] == items[i]["author"] for i in test])
            print(f"  week {wk} held out: {len(test)} photos, accuracy {acc:.3f}", flush=True)
        report["leave_one_week_out"] = metrics(confusion(held_pred, items, cred, authors), authors)
        print(f"LOWO accuracy: {report['leave_one_week_out']['accuracy']:.4f}", flush=True)

    # 5. outputs
    rows = []
    for i in anon:
        (a1, (s1, o1)), (a2, (s2, _)) = best2(S[i])
        margin = s1 - s2
        conf = "high" if margin >= 0.004 else ("medium" if margin >= 0.0015 else "contested")
        rows.append({"filename": items[i]["name"], "week": items[i]["week"], "predicted_author": a1, "confidence": conf,
                     "top1_score": round(s1, 6), "top2_score": round(s2, 6), "top2_author": a2, "margin": round(margin, 6),
                     "sensor_orientation": o1, "evidence_method": "PRNU wavelet residual, orientation-aware, semi-supervised"})
    with open(os.path.join(args.output, "japan_attribution.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    with open(os.path.join(args.output, "japan_toorisugari_candidates.csv"), "w", newline="", encoding="utf-8") as fh:
        tr = [r for r in rows if r["predicted_author"] == "toorisugari"]
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(tr)
    full = {items[i]["name"]: {a: {"score": round(S[i][a][0], 6), "orientation": S[i][a][1]} for a in authors} for i in anon}
    json.dump(full, open(os.path.join(args.output, "japan_scores_all_authors.json"), "w"), indent=1)
    counts = {}
    for r in rows:
        counts.setdefault(r["predicted_author"], {}).setdefault(r["week"], 0)
        counts[r["predicted_author"]][r["week"]] += 1
    report["anonymous_counts_by_author_and_week"] = counts
    report["credited_orientation"] = {items[i]["name"] + "@" + items[i]["author"]: orient[i] for i in cred}
    report["runtime_seconds"] = round(time.time() - t0, 1)
    json.dump(report, open(os.path.join(args.output, "japan_validation.json"), "w"), indent=1)
    for a in authors:
        np.save(os.path.join(cache, f"K_{a}.npy"), K[a].astype(np.float32))
    print(json.dumps(counts, indent=0))
    print(f"done in {time.time() - t0:.0f}s -> {args.output}")


if __name__ == "__main__":
    main()
