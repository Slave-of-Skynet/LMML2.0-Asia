"""Candidates and page decisions from the JPEG-history maps (jpeg_forensics.py).

Exhibits (honest, out-of-fold maps only):
  * pool recall of the top-K JPEG peaks (K = 1, 2, 3, 5, 8) per stamp,
  * the same for TruFor out-of-fold maps (``outputs/cv/exhibit_oof``) and for
    a fixed, untuned fusion ``jpeg + 0.5 * trufor`` (weight fixed a priori,
    reported, not selected).

Albums:
  * per stamp the top-K NMS peaks of the JPEG model map (``*_model.npy``),
    values comparable across stamps because one model scores all of them;
  * ``candidates.csv`` in the format of ``rescore.py`` (album, page, rank,
    x, y, stamp, p, prior, area, j, j_free, t) so ``read_letters.py
    --rescored`` can read it; ``prior`` = odds normalisation as in rescore;
  * ``boxes.csv`` - the page-level argmax (one row per page, absolute
    1600x1200 coordinates), never letter-derived;
  * ``panel_<album>.png`` - automatic dot panel of the 26 page argmaxes.

No album label, original stamp, word list or manual coordinate is used.
"""
import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

from geometry import exhibit_box, stamps
from jpeg_forensics import MARGIN, nms_peaks

ODDS_MISS = 0.35
FUSE_W = 0.5


def exhibit_eval(data, jdir, cvdir, ks=(1, 2, 3, 5, 8)):
    res = {}
    rows = {}
    for f in sorted((data / "exhibits").glob("exhibit_??.jpg")):
        image = cv2.imread(str(f))
        x, y, w, h = exhibit_box(image)
        mask = cv2.imread(str(f.with_name(f.stem + "_mask.png")), 0)[y:y + h, x:x + w] > 127
        dil = cv2.dilate(mask.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
        valid = np.zeros(mask.shape, bool)
        valid[MARGIN:-MARGIN, MARGIN:-MARGIN] = True
        maps = {"jpeg": np.load(jdir / "exhibit_oof" / f"{f.stem}.npy").astype(np.float32),
                "jpeg_free": np.load(jdir / "exhibit_free" / f"{f.stem}.npy").astype(np.float32)}
        tp = cvdir / "exhibit_oof" / f"{f.stem}.npy"
        if tp.exists():
            maps["trufor"] = np.load(tp).astype(np.float32)
            maps["fused"] = maps["jpeg"] + FUSE_W * maps["trufor"]
        rows[f.stem] = {}
        for name, heat in maps.items():
            pk = nms_peaks(heat, valid, max(ks))
            hit = [bool(dil[py, px]) for px, py, _ in pk]
            first = hit.index(True) + 1 if True in hit else 0
            rows[f.stem][name] = first
    for name in next(iter(rows.values())):
        res[name] = {f"top{k}": int(sum(0 < r[name] <= k for r in rows.values())) for k in ks}
    return {"summary": res, "rank_of_first_hit": rows}


def album_candidates(data, jdir, cvdir, topk, folds=(0, 1, 2, 3, 4)):
    out = []
    for path in sorted((data / "albums").glob("*/*.jpg")):
        album, page = path.parent.name, path.stem
        key = f"{album}_{page}"
        jm = np.load(jdir / "album_maps" / f"{key}_model.npy").astype(np.float32)
        jf = np.load(jdir / "album_maps" / f"{key}_free.npy").astype(np.float32)
        tm = None
        tpaths = [cvdir / "album_folds" / f"{key}_f{k}.npy" for k in folds]
        if all(p.exists() for p in tpaths):
            tm = cv2.resize(np.mean([np.load(p).astype(np.float32) for p in tpaths], 0),
                            (jm.shape[1], jm.shape[0]), interpolation=cv2.INTER_LINEAR)
        image_shape = jm.shape
        regions = stamps(cv2.imread(str(path)))
        area = float(sum((w - 12) * (h - 12) for _, _, w, h in regions))
        cands = []
        for si, (bx, by, bw, bh) in enumerate(regions):
            x, y, w, h = bx + 6, by + 6, bw - 12, bh - 12
            sub = jm[y:y + h, x:x + w]
            valid = np.zeros(sub.shape, bool)
            valid[MARGIN:-MARGIN, MARGIN:-MARGIN] = True
            valid &= np.isfinite(sub)
            for px, py, v in nms_peaks(np.nan_to_num(sub, nan=-1), valid, topk):
                X, Y = x + px, y + py
                t = float(tm[max(0, Y - 16):Y + 17, max(0, X - 22):X + 23].max()) if tm is not None else 0.0
                cands.append({"album": album, "page": page, "stamp": si, "x": X, "y": Y,
                              "j": v, "j_free": float(jf[Y, X]), "t": t, "area": area})
        for c in cands:
            c["p"] = float(np.clip(c["j"], 1e-4, 1 - 1e-4))
        cands.sort(key=lambda c: -c["j"])
        odds = np.array([c["p"] / (1 - c["p"]) for c in cands])
        prior = odds / (odds.sum() + ODDS_MISS)
        for r, (c, pr) in enumerate(zip(cands, prior)):
            c["rank"], c["prior"] = r, float(pr)
        out += cands
        assert image_shape == (1200, 1600)
    return out


def agreement(tops, pseudo_path, rescored_path, dx=48, dy=36):
    """Page argmax vs the independent re-noise detector's single-strong pages."""
    if not pseudo_path.exists():
        return None
    pseudo = {(q["album"], q["page"]): q for q in json.loads(pseudo_path.read_text())}
    near = lambda c, q: abs(c["x"] - q["x"]) <= dx and abs(c["y"] - q["y"]) <= dy
    res = {"pages": len(pseudo)}
    jt = {(c["album"], c["page"]): c for c in tops}
    res["jpeg_top1"] = sum(near(jt[k], q) for k, q in pseudo.items() if k in jt)
    if rescored_path.exists():
        best_t, best_p = {}, {}
        for r in csv.DictReader(open(rescored_path, newline="", encoding="utf-8")):
            k = (r["album"], r["page"])
            c = {"x": float(r["x"]), "y": float(r["y"]), "t": float(r["t"]), "p": float(r["p"])}
            if k not in best_t or c["t"] > best_t[k]["t"]:
                best_t[k] = c
            if k not in best_p or c["p"] > best_p[k]["p"]:
                best_p[k] = c
        res["trufor_top1"] = sum(near(best_t[k], q) for k, q in pseudo.items() if k in best_t)
        res["rescored_top1"] = sum(near(best_p[k], q) for k, q in pseudo.items() if k in best_p)
    return res


def panel(points, path, title):
    img = np.full((600, 800, 3), 255, np.uint8)
    for x, y, s in points:
        cv2.circle(img, (int(x / 2), int(y / 2)), int(3 + 6 * s), (0, 0, 200), -1)
    cv2.putText(img, title, (10, 25), 0, 0.7, (0, 0, 0), 2)
    cv2.imwrite(str(path), img)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--jpeg", type=Path, default=Path("outputs/jpeg"))
    parser.add_argument("--cv", type=Path, default=Path("outputs/cv"))
    parser.add_argument("--output", type=Path, default=Path("outputs/jpeg_cands"))
    parser.add_argument("--topk", type=int, default=4)
    parser.add_argument("--pseudo", type=Path, default=Path("outputs/noise_pseudo.json"),
                        help="noise_forensics.py single-strong pages (independent agreement check)")
    parser.add_argument("--rescored", type=Path, default=Path("outputs/rescore_tk12/rescored_candidates.csv"),
                        help="old TruFor/rescore pool, for the comparison only")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    report = {"exhibits": exhibit_eval(args.data, args.jpeg, args.cv)}
    print(json.dumps(report["exhibits"]["summary"], indent=1), flush=True)
    cands = album_candidates(args.data, args.jpeg, args.cv, args.topk)
    fields = ["album", "page", "rank", "x", "y", "stamp", "p", "prior", "area", "j", "j_free", "t"]
    with (args.output / "candidates.csv").open("w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        wr.writeheader()
        for c in cands:
            wr.writerow({k: (round(v, 5) if isinstance(v, float) else v) for k, v in c.items()})
    tops = [c for c in cands if c["rank"] == 0]
    with (args.output / "boxes.csv").open("w", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh)
        wr.writerow(["album", "page", "x", "y", "w", "h", "stamp", "score"])
        for c in tops:
            wr.writerow([c["album"], c["page"], c["x"] - 32, c["y"] - 24, 64, 48, c["stamp"], round(c["j"], 4)])
    for album in sorted({c["album"] for c in tops}):
        pts = [(c["x"], c["y"], c["j"]) for c in tops if c["album"] == album]
        panel(pts, args.output / f"panel_{album}.png", f"{album} JPEG top-1 (n={len(pts)})")
    margins = []
    for key in sorted({(c["album"], c["page"]) for c in cands}):
        page = sorted([c for c in cands if (c["album"], c["page"]) == key], key=lambda c: -c["j"])
        other = [c for c in page[1:] if c["stamp"] != page[0]["stamp"]]
        margins.append(page[0]["j"] - (other[0]["j"] if other else 0))
    report["albums"] = {"pages": len(tops), "unique_pages": len({(c["album"], c["page"]) for c in tops}),
                        "top_score_quantiles": [float(v) for v in np.percentile([c["j"] for c in tops], [10, 25, 50, 75, 90])],
                        "cross_stamp_margin_quantiles": [float(v) for v in np.percentile(margins, [10, 25, 50, 75, 90])]}
    report["albums"]["agreement_with_strong_noise_pages"] = agreement(tops, args.pseudo, args.rescored)
    print(json.dumps(report["albums"], indent=1))
    (args.output / "report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
