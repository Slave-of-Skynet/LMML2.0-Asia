#!/usr/bin/env python3
"""Read the hidden word from the captions of Toorisugari's photos.

Mechanism (organizer hint): treat every caption line as a row of character cells counted from the left.
For each week 8-12 compare the mean "density" of each cell position in Toorisugari's photos with the mean over
all photos of that week (kanji are denser than kana). Stacking the five weekly rows draws letters.

Steps
  1. Caption geometry per photo (subcells.process): caption box (opaque or translucent), glyph mask, text lines,
     character pitch = line spacing / 1.35 snapped to the generator font sizes (26/32/38 px), cells aligned to
     the common left edge of the caption block.
  2. Cell density, two independent measures:
       a) pixel measures: binary ink fraction, per-photo normalised ink, mean cell brightness;
       b) kanji probability from a logistic classifier trained on 25 hand-read captions (1199 cells, 5-fold CV
          accuracy 0.954; labels in cellclf.LABELLED).
  3. Week matrix W[w, k] = mean_k(Toorisugari, week w) - mean_k(all photos, week w), z-scored against random
     groups of the same size from the same week.
  4. Template decoding with a 3x5 pixel font (M is 5 columns wide), plus a selection-corrected significance test
     (best word found in Toorisugari vs best word found in 300 random photo groups).

Usage: python decode_secret.py --dataset <dataset> --attribution <japan_attribution.csv> --out <dir>
"""
import argparse, csv, json, os, pickle, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--attribution", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--target", default="toorisugari")
    args = ap.parse_args()
    os.environ["JAPAN_DATASET"] = args.dataset
    os.makedirs(args.out, exist_ok=True)
    import subcells, cellclf, font_decode as FD
    subcells.DATA = args.dataset; cellclf.subcells.DATA = args.dataset
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    attr = {r["filename"]: r["predicted_author"] for r in csv.DictReader(open(args.attribution, encoding="utf-8"))}

    # 1-2a. geometry and pixel measures
    cells = {}
    for n in sorted(attr):
        try:
            cells[n] = subcells.process(os.path.join(args.dataset, "anonymous", n))
        except Exception:
            cells[n] = None
    print("caption geometry: %d/%d photos" % (sum(v is not None for v in cells.values()), len(cells)), flush=True)

    # 2b. kanji classifier
    X, Y = [], []
    for n, txt in cellclf.LABELLED.items():
        rows = cellclf.cell_images(os.path.join(args.dataset, "anonymous", n + ".jpg"))
        lines = txt.split("|")
        if rows is None or len(rows) != len(lines):
            continue
        for r, l in zip(rows, lines):
            if len(r) == len(l):
                for v, c in zip(r, l):
                    X.append(cellclf.feats(v)); Y.append(cellclf.cls_of(c))
    X, Y = np.array(X), np.array(Y)
    clf = LogisticRegression(max_iter=3000, C=0.5)
    cv = float(cross_val_score(clf, X, Y, cv=5).mean())
    clf.fit(X, Y)
    print("cell classifier: %d labelled cells, 5-fold CV accuracy %.4f" % (len(Y), cv), flush=True)
    kprob = {}
    for n in sorted(attr):
        try:
            rows = cellclf.cell_images(os.path.join(args.dataset, "anonymous", n))
        except Exception:
            rows = None
        kprob[n] = [clf.predict_proba(np.array([cellclf.feats(v) for v in r]))[:, 2].tolist() if len(r) else [] for r in rows] if rows else []

    L = 20
    weeks = range(8, 13)
    wk = {w: [n for n in attr if int(n[1:3]) == w] for w in weeks}
    rng = np.random.default_rng(1)

    def series(measure):
        out = {}
        for n in attr:
            if measure == "kanji":
                out[n] = [np.array(l) for l in kprob[n]]
                continue
            d = cells[n]
            if not d:
                out[n] = []; continue
            key = {"ink": "lines", "inknorm": "lines", "cellmean": "cellmean", "cellnorm": "cellmean"}[measure]
            ls = [np.array(l, float) for l in d[key]]
            if measure.endswith("norm"):
                a = np.concatenate(ls); a = a[np.isfinite(a)]
                ls = [l / a[a > 0.02].mean() for l in ls] if key == "lines" else [l - np.median(a) for l in ls]
            out[n] = ls
        return out

    def prof(V, names):
        s = np.zeros(L); c = np.zeros(L)
        for n in names:
            for l in V[n]:
                for k, x in enumerate(l[:L]):
                    if np.isfinite(x):
                        s[k] += x; c[k] += 1
        return s / np.maximum(c, 1)

    def zmatrix(V):
        Z = []
        for w in weeks:
            tw = [n for n in wk[w] if attr[n] == args.target]
            pa = prof(V, wk[w]); d = prof(V, tw) - pa
            null = np.array([prof(V, list(rng.choice(wk[w], len(tw), replace=False))) - pa for _ in range(400)])
            Z.append((d - null.mean(0)) / (null.std(0) + 1e-9))
        return np.array(Z)

    mats = {m: zmatrix(series(m)) for m in ("ink", "inknorm", "cellmean", "cellnorm", "kanji")}
    Zpix = np.mean([mats[m] for m in ("ink", "inknorm", "cellmean", "cellnorm")], 0)
    Zall = np.mean([Zpix, mats["kanji"]], 0)
    np.savez(os.path.join(args.out, "week_matrices.npz"), Zpix=Zpix, Zall=Zall, **mats)

    # 4. template decoding
    M5 = np.array([[int(c) for c in r] for r in ["10001", "11011", "10101", "10001", "10001"]], float)
    WIDE = {"M": M5}

    def render(word):
        parts = []
        for i, ch in enumerate(word):
            if i:
                parts.append(np.zeros((5, 1)))
            parts.append(WIDE.get(ch, FD.T[ch]))
        return np.hstack(parts)

    def corr(Xm, B):
        x = Xm - Xm.mean(); b = B - B.mean()
        return float((x * b).sum() / (np.linalg.norm(x) * np.linalg.norm(b) + 1e-12))

    def best_word(R):
        best = (-1, None, None)
        letters = list(FD.T)
        for off in range(3):
            for wide_slot in (None, 2):
                pos, word, ok = off, "", True
                for slot in range(4):
                    wd = 5 if slot == wide_slot else 3
                    if pos + wd > L:
                        ok = False; break
                    blk = R[:, pos:pos + wd]; x = blk - blk.mean()
                    pool = ["M"] if wd == 5 else letters
                    tmpl = lambda ch: WIDE.get(ch, FD.T[ch])
                    ch = max(pool, key=lambda c: (x * (tmpl(c) - tmpl(c).mean())).sum() / (np.linalg.norm(tmpl(c) - tmpl(c).mean()) + 1e-9))
                    word += ch; pos += wd + 1
                if ok:
                    B = render(word); s = corr(R[:, off:off + B.shape[1]], B)
                    if s > best[0]:
                        best = (s, word, off)
        return best

    report = {"classifier_cv_accuracy": cv}
    for name, R in (("pixel", Zpix), ("kanji", mats["kanji"]), ("combined", Zall)):
        s, word, off = best_word(np.nan_to_num(R[:, :L]))
        report[name] = {"best_word": word, "offset": off, "score": round(s, 4)}
        print(f"{name:9s} best word {word} (offset {off}) score {s:.3f}", flush=True)

    # selection-corrected null: random groups of the same size per week, same search
    V = series("kanji"); Vp = {m: series(m) for m in ("ink", "inknorm", "cellmean", "cellnorm")}
    nb = []
    for _ in range(200):
        groups = {w: list(rng.choice(wk[w], sum(attr[n] == args.target for n in wk[w]), replace=False)) for w in weeks}
        R = np.array([prof(V, groups[w]) - prof(V, wk[w]) for w in weeks])
        R = R / (np.abs(R).max() + 1e-9)
        nb.append(best_word(R)[0])
    Rk = np.array([prof(V, [n for n in wk[w] if attr[n] == args.target]) - prof(V, wk[w]) for w in weeks])
    st = best_word(Rk / (np.abs(Rk).max() + 1e-9))
    nb = np.array(nb)
    report["significance_kanji_rate"] = {"target_best": [round(st[0], 4), st[1]], "null_mean": round(float(nb.mean()), 4),
                                         "null_std": round(float(nb.std()), 4), "null_max": round(float(nb.max()), 4),
                                         "p_value_upper": float((np.sum(nb >= st[0]) + 1) / (len(nb) + 1))}
    print("selection-corrected test:", report["significance_kanji_rate"], flush=True)
    json.dump(report, open(os.path.join(args.out, "secret_report.json"), "w"), indent=1)

    from PIL import Image, ImageDraw
    B = render(report["combined"]["best_word"]); B = np.hstack([B, np.zeros((5, L - B.shape[1]))])
    c = 40; im = Image.new("L", (L * c + 80, 3 * (5 * c + 46)), 255); d = ImageDraw.Draw(im)
    panels = [(Zpix / 2.0, "Toorisugari minus week: z-score of pixel cell darkness (dark = denser)"),
              (mats["kanji"] / 2.0, "Toorisugari minus week: z-score of kanji probability"), (B * 2 - 1, "Best fitted template")]
    for k, (Mx, title) in enumerate(panels):
        y0 = k * (5 * c + 46) + 24; d.text((10, y0 - 20), title, fill=0)
        for i in range(5):
            d.text((10, y0 + i * c + 14), f"w{8 + i}", fill=0)
            for j in range(L):
                v = float(np.clip(np.nan_to_num(Mx[i, j]), -1, 1))
                d.rectangle([60 + j * c, y0 + i * c, 60 + (j + 1) * c - 2, y0 + (i + 1) * c - 2], fill=int(128 - 127 * v))
        for j in range(L):
            d.text((60 + j * c + 14, y0 + 5 * c + 2), str(j), fill=0)
    im.save(os.path.join(args.out, "secret_matrix.png"))


if __name__ == "__main__":
    main()
