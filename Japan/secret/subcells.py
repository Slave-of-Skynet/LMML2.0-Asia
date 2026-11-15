import os
"""Subtitle character-cell densities per photo.
For each photo: find the caption block, split into text lines, estimate the character pitch (monospaced
full-width Japanese), and measure the ink density of every cell counted from the left edge of the block."""
import numpy as np, os, sys, json
from scipy import ndimage
from PIL import Image
from concurrent.futures import ProcessPoolExecutor

DATA = os.environ.get("JAPAN_DATASET", "dataset")
FONT_PITCHES = (26.0, 32.0, 38.0)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def glyph_mask(x):
    g = x.mean(-1)
    neutral = (x.max(-1) - x.min(-1)) < 22
    bg = ndimage.grey_opening(g, size=(15, 15))
    med = ndimage.median_filter(g[::2, ::2], 11).repeat(2, 0).repeat(2, 1)[: g.shape[0], : g.shape[1]]
    dark = ndimage.binary_erosion(med < 70, np.ones((9, 9)))
    return (g - bg > 45) & neutral & (g > 100) & dark


def caption_region(x, gm):
    import mask2
    blk = ndimage.binary_closing(x.max(-1) < 3, np.ones((25, 25)))
    cands = []
    lab, n = ndimage.label(blk)
    for k, sl in enumerate(ndimage.find_objects(lab), 1):
        if (lab[sl] == k).sum() > 3000:
            cands.append((sl[0].start, sl[0].stop, sl[1].start, sl[1].stop))
    cands += mask2.translucent_boxes(x)
    best, bs = None, 0
    for (y0, y1, x0, x1) in cands:
        s = gm[y0:y1, x0:x1].sum()
        if s > bs:
            bs, best = s, (y0, y1, x0, x1)
    return best


def lines_of(gm):
    prof = gm.sum(1)
    on = prof > max(3, 0.05 * prof.max())
    on = ndimage.binary_closing(on, np.ones(5))
    lab, n = ndimage.label(on)
    out = []
    for sl in ndimage.find_objects(lab):
        y0, y1 = sl[0].start, sl[0].stop
        if y1 - y0 >= 10:
            out.append((y0, y1))
    return out


def process(path):
    x = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)
    gm = glyph_mask(x)
    reg = caption_region(x, gm)
    if reg is None:
        return None
    y0, y1, x0, x1 = reg
    y0 += 3; x0 += 3; y1 -= 3; x1 -= 3
    gimg = x.mean(-1)[y0:y1, x0:x1]
    bgi = ndimage.grey_opening(gimg, size=(15, 15))
    gsub = gm[y0:y1, x0:x1]
    tcol = np.percentile(gimg[gsub], 90) if gsub.any() else 255.0
    inkc = np.clip((gimg - bgi) / max(tcol - np.median(bgi), 1.0), 0, 1)
    sub = ndimage.binary_opening(gm[y0:y1, x0:x1], np.ones((2, 2))) | (gm[y0:y1, x0:x1] & ndimage.binary_dilation(ndimage.binary_opening(gm[y0:y1, x0:x1], np.ones((2, 2))), np.ones((3, 3))))
    lines = [l for l in lines_of(sub) if 12 <= l[1] - l[0] <= 80]
    if not lines:
        return None
    hs = [b - a for a, b in lines]
    em = float(np.median(hs))  # glyph-ink height ~ font em
    # refine pitch with column autocorrelation of all lines
    info = []
    for (a_, b_) in lines:
        cols = np.where(sub[a_:b_].any(0))[0]
        if len(cols):
            info.append((a_, b_, int(cols[0]), int(cols[-1])))
    if not info:
        return None
    p0 = float(np.median([b_ - a_ for a_, b_, _, _ in info])) / 0.87
    st = np.array([c0 for _, _, c0, _ in info], float)
    cnt = [(np.abs(st - s0) < 0.5 * p0).sum() for s0 in st]
    s_best = st[int(np.argmax(cnt))]
    info = [t for t in info if abs(t[2] - s_best) < 0.5 * p0]
    left = int(min(t[2] for t in info))
    if len(info) >= 2:
        cen = np.array([(a_ + b_) / 2 for a_, b_, _, _ in info])
        sp = np.diff(cen)
        sp = sp[(sp > 0.9 * 1.35 * 0.8 * p0) & (sp < 1.6 * 1.35 * p0)]
        best_p = float(np.median(sp)) / 1.35 if len(sp) else p0
    else:
        best_p = p0
    # refine: every line width should be close to an integer number of cells
    widths = np.array([c1 - left + 1 for a, b, c0, c1 in info], float)
    cand = np.arange(0.96 * best_p, 1.04 * best_p, 0.05)
    sc = [np.cos(2 * np.pi * (widths + 0.08 * p) / p).sum() for p in cand]
    pitch = float(cand[int(np.argmax(sc))])
    pitch = min(FONT_PITCHES, key=lambda q: abs(q - pitch))
    res = []; bright = []; dark = []; inks = []
    for a, b, c0, c1 in info:
        n = int(round((c1 - left + 1 + 0.08 * pitch) / pitch))
        dens = []; br = []; dk = []; ik = []
        for k in range(n):
            xa = int(round(left + k * pitch)); xb = int(round(left + (k + 1) * pitch))
            cell = sub[a:b, xa:xb]
            gi = gimg[a:b, xa:xb]
            dens.append(float(cell.mean()) if cell.size else 0.0)
            br.append(float(gi[cell].mean()) if cell.any() else float('nan'))
            dk.append(float(gi.mean()) if gi.size else float('nan'))
            ik.append(float(inkc[a:b, xa:xb].mean()) if gi.size else float('nan'))
        res.append(dens); bright.append(br); dark.append(dk); inks.append(ik)
    return dict(box=[int(v) for v in reg], em=em, pitch=float(pitch), left=left, lines=res, bright=bright, cellmean=dark, ink=inks, widths=[int(c1 - left + 1) for a, b, c0, c1 in info], starts=[int(c0 - left) for a, b, c0, c1 in info])


def job(rel):
    try:
        return rel, process(os.path.join(DATA, rel))
    except Exception as e:
        return rel, {"error": str(e)}


