"""Training-free detector for one family of alterations: re-noised patches.

Observation on the labelled exhibits (no album page was inspected): roughly a
third of the altered rectangles carry *synthetic* noise - white in space and
independent between the three colour channels - whereas the genuine grain of
a down-scaled stamp scan is low-pass and strongly correlated across channels.
Two statistics isolate that signature:

* ``independent_noise_map`` - box mean of the high-pass colour-difference
  energy, damped by the local cross-channel correlation of the high-pass
  residual (mean-based, sensitive, used to place the peak);
* ``robust_ratio_maps``     - local *medians* of |HF colour difference| and
  |HF luminance|; their ratio is robust to glyph edges inside the patch and is
  expressed as a per-stamp z-score ``zr`` against the stamp's own
  distribution.

A stamp peak is STRONG when ``zr >= 1.6`` and ``conf >= 1.7`` and the blob
around it is a plausible filled rectangle (fill >= 0.6, 22-115 x 20-95 px);
on the 30 exhibits that gives hits with 0 false peaks, but the thresholds
were chosen on those same 30 images, so the zero is optimistic: pages with
two "strong" stamps (impossible if both were true) put the false-strong rate
of an unaltered stamp at a few per cent.  WEAK: ``zr >= 1.2`` and
``conf >= 1.4``, or strong statistics with an implausible shape.  The detector says nothing about the other alteration
families (clean erasures, digit copies without added noise).

No original stamp, catalogue, duplicate or manual album coordinate is used.
"""
import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

from geometry import stamps

KERNEL = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float32) / 4
STRONG = {"zr": 1.6, "conf": 1.7}
WEAK = {"zr": 1.2, "conf": 1.4}
MARGIN = 22          # perforation / frame margin excluded inside each stamp
BOX = (41, 29)       # (width, height) of the averaging box, ~ median edit size
MEDIAN = 25          # median window of the robust maps
FIELDS = ["album", "page", "stamp", "rank", "x", "y", "conf", "zr", "zc",
          "ratio", "tier", "fill", "bx", "by", "bw", "bh"]


def _highpass(image):
    g = image.astype(np.float32)
    return [cv2.filter2D(g[..., c], -1, KERNEL) for c in range(3)]


def independent_noise_map(image, box=BOX):
    kb, kg, kr = _highpass(image)
    chroma = (kb - kg) ** 2 + (kr - kg) ** 2 + (kb - kr) ** 2
    cov = kb * kg + kg * kr + kb * kr
    var = kb * kb + kg * kg + kr * kr
    e_c = cv2.blur(chroma, box)
    rho = cv2.blur(cov, box) / (cv2.blur(var, box) + 0.5)
    return e_c * np.clip(1 - rho, 0, 2)


def _median(a, k, scale=12):
    q = np.clip(a * scale, 0, 255).astype(np.uint8)
    return cv2.medianBlur(q, k).astype(np.float32) / scale


def robust_ratio_maps(image, k=MEDIAN):
    kb, kg, kr = _highpass(image)
    chroma = (np.abs(kb - kg) + np.abs(kr - kg) + np.abs(kb - kr)) / 3
    luma = np.abs(kb + kg + kr) / 3
    m_c, m_l = _median(chroma, k), _median(luma, k)
    return m_c, m_l, m_c / (m_l + 0.15)


def tier_of(cand):
    if cand["zr"] >= STRONG["zr"] and cand["conf"] >= STRONG["conf"]:
        return "strong"
    if cand["zr"] >= WEAK["zr"] and cand["conf"] >= WEAK["conf"]:
        return "weak"
    return "none"


def _patch_box(ratio_fine, px, py):
    """Extent and rectangularity of the re-noised blob around page point.

    ``ratio_fine`` is the robust ratio map with a small median window.  The
    blob is the connected region above the midpoint between the local level
    and the neighbourhood background.  Returns (x, y, w, h, fill) in page
    coordinates or None; ``fill`` = blob area / bounding-box area (genuine
    edits are filled axis-aligned rectangles: 0.64-0.89 on the exhibits).
    """
    y0, x0 = max(0, py - 80), max(0, px - 110)
    r = ratio_fine[y0:py + 81, x0:px + 111]
    cy, cx = py - y0, px - x0
    level = float(np.median(r[max(0, cy - 8):cy + 9, max(0, cx - 10):cx + 11]))
    binary = (r > (level + float(np.median(r))) / 2).astype(np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    _, labels, stats, _ = cv2.connectedComponentsWithStats(binary)
    lab = labels[cy, cx]
    if lab == 0:
        return None
    x, y, w, h, area = (int(v) for v in stats[lab])
    return x0 + x, y0 + y, w, h, area / float(w * h)


def shape_ok(box):
    return box is not None and box[4] >= 0.6 and 22 <= box[2] <= 115 and 20 <= box[3] <= 95


def detect_page(image, topn=2, regions=None):
    """Return ranked re-noised-patch candidates for every stamp of a page."""
    smap = independent_noise_map(image)
    m_c, _, ratio = robust_ratio_maps(image)
    ratio_fine = robust_ratio_maps(image, k=15)[2]
    out = []
    for si, (x, y, w, h) in enumerate(regions or stamps(image)):
        sl = (slice(y + MARGIN, y + h - MARGIN), slice(x + MARGIN, x + w - MARGIN))
        sub = smap[sl].copy()
        if sub.size == 0:
            continue
        rs, mc = ratio[sl], m_c[sl]
        p90 = float(np.percentile(sub, 90))
        r50, r95 = (float(v) for v in np.percentile(rs, [50, 95]))
        c50, c95 = (float(v) for v in np.percentile(mc, [50, 95]))
        for rank in range(topn):
            cy, cx = (int(v) for v in np.unravel_index(sub.argmax(), sub.shape))
            peak = float(sub[cy, cx])
            if peak < 0:
                break
            y0, y1, x0, x1 = max(0, cy - 10), cy + 11, max(0, cx - 14), cx + 15
            rr = float(np.median(rs[y0:y1, x0:x1]))
            cc = float(np.median(mc[y0:y1, x0:x1]))
            cand = {
                "stamp": si, "rank": rank,
                "x": x + MARGIN + cx, "y": y + MARGIN + cy,
                "conf": float(np.log((peak + 0.05) / (p90 + 0.05))),
                "zr": (rr - r50) / (r95 - r50 + 1e-3),
                "zc": (cc - c50) / (c95 - c50 + 1e-3),
                "ratio": rr,
            }
            cand["tier"] = tier_of(cand)
            box = _patch_box(ratio_fine, cand["x"], cand["y"]) if cand["tier"] != "none" else None
            cand["fill"] = box[4] if box is not None else 0.0
            if cand["tier"] == "strong" and not shape_ok(box):
                cand["tier"] = "weak"       # strong statistics, implausible shape
            if box is None:
                box = (cand["x"] - 32, cand["y"] - 20, 64, 40, 0.0)
            cand["bx"], cand["by"], cand["bw"], cand["bh"] = box[:4]
            out.append(cand)
            sub[max(0, cy - 45):cy + 46, max(0, cx - 60):cx + 61] = -1
    return out


def point_features(image, regions, points):
    """Noise statistics at arbitrary page points (for candidate rescoring).

    Returns one dict per point with ``conf`` and ``zr`` measured exactly as in
    ``detect_page`` but at the given location instead of the stamp maximum.
    Points outside every stamp margin get zeros.
    """
    smap = independent_noise_map(image)
    m_c, _, ratio = robust_ratio_maps(image)
    kb, kg, kr = _highpass(image)
    luma_hf = np.abs(kb + kg + kr) / 3
    stats = []
    for x, y, w, h in regions:
        sl = (slice(y + MARGIN, y + h - MARGIN), slice(x + MARGIN, x + w - MARGIN))
        if smap[sl].size == 0:
            stats.append(None)
            continue
        r50, r95 = (float(v) for v in np.percentile(ratio[sl], [50, 95]))
        stats.append((float(np.percentile(smap[sl], 90)), r50, r95))
    out = []
    for px, py in points:
        feat = {"conf": 0.0, "zr": 0.0, "flat": 0.0, "stamp": -1}
        for si, (x, y, w, h) in enumerate(regions):
            if x <= px < x + w and y <= py < y + h and stats[si] is not None:
                p90, r50, r95 = stats[si]
                cx = int(np.clip(px, x + MARGIN, x + w - MARGIN - 1))
                cy = int(np.clip(py, y + MARGIN, y + h - MARGIN - 1))
                y0, y1 = max(y + MARGIN, cy - 14), min(y + h - MARGIN, cy + 15)
                x0, x1 = max(x + MARGIN, cx - 20), min(x + w - MARGIN, cx + 21)
                peak = float(smap[y0:y1, x0:x1].max())
                rr = float(np.median(ratio[max(y0, cy - 10):cy + 11, max(x0, cx - 14):cx + 15]))
                # "noise hole": grain inside a nominal edit box vs. its ring
                inner = luma_hf[max(y, cy - 14):cy + 15, max(x, cx - 24):cx + 25]
                ring = luma_hf[max(y, cy - 38):min(y + h, cy + 39),
                               max(x, cx - 56):min(x + w, cx + 57)].copy()
                ry, rx = min(cy - max(y, cy - 38), 38), min(cx - max(x, cx - 56), 56)
                ring[max(0, ry - 22):ry + 23, max(0, rx - 34):rx + 35] = np.nan
                flat = float(np.log((np.nanmedian(ring) + 0.1) / (np.median(inner) + 0.1)))
                feat = {"conf": float(np.log((peak + 0.05) / (p90 + 0.05))),
                        "zr": (rr - r50) / (r95 - r50 + 1e-3), "flat": flat, "stamp": si}
                break
        out.append(feat)
    return out


def page_decision(cands):
    """Best strong candidate if exactly one stamp is strong, else None."""
    strong = [c for c in cands if c["tier"] == "strong"]
    if not strong or len({c["stamp"] for c in strong}) != 1:
        return None
    return max(strong, key=lambda c: c["conf"])


def check_exhibits(data):
    report = {"strong_hits": 0, "strong_false": 0, "weak_hits": 0, "weak_false": 0}
    for image_path in sorted((data / "exhibits").glob("exhibit_??.jpg")):
        image = cv2.imread(str(image_path))
        mask = cv2.imread(str(image_path.with_name(image_path.stem + "_mask.png")), 0)
        for cand in detect_page(image):
            if cand["tier"] == "none":
                continue
            hit = bool(mask[cand["y"], cand["x"]])
            report[f"{cand['tier']}_{'hits' if hit else 'false'}"] += 1
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True,
                        help="CSV of all per-stamp candidates")
    parser.add_argument("--pseudo", type=Path,
                        help="optional JSON of single-strong-stamp pages")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    print("exhibit check:", json.dumps(check_exhibits(args.data)))
    rows, pseudo = [], []
    for album in sorted(p for p in (args.data / "albums").iterdir() if p.is_dir()):
        for page in sorted(album.glob("*.jpg")):
            image = cv2.imread(str(page))
            regions = stamps(image)
            cands = detect_page(image, regions=regions)
            for cand in cands:
                rows.append({"album": album.name, "page": page.stem, **cand})
            best = page_decision(cands)
            if best is not None:
                pseudo.append({"album": album.name, "page": page.stem,
                               "stamp": best["stamp"], "x": best["x"], "y": best["y"],
                               "box": [best["bx"], best["by"], best["bw"], best["bh"]],
                               "stamps": [list(r) for r in regions]})
    with args.output.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: (round(v, 4) if isinstance(v, float) else v)
                             for k, v in row.items()})
    if args.pseudo:
        args.pseudo.write_text(json.dumps(pseudo, indent=1), encoding="utf-8")
    tiers = {t: sum(r["tier"] == t for r in rows) for t in ("strong", "weak")}
    print(f"wrote {len(rows)} candidates; tiers {tiers}; "
          f"{len(pseudo)} single-strong pages")


if __name__ == "__main__":
    main()
