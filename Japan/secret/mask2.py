import numpy as np, os, sys
from scipy import ndimage
import feat


def _runs(row_bool, minlen):
    r = np.flatnonzero(np.diff(np.r_[0, row_bool.astype(np.int8), 0]))
    return [(a, b) for a, b in zip(r[::2], r[1::2]) if b - a >= minlen]


def translucent_boxes(x, thr=0.45, minlen=160):
    L = np.log(x.mean(-1) + 4.0)
    d = np.zeros_like(L)
    d[3:-3] = L[:-6] - L[6:]          # >0: brighter above, darker below (top edge of dark box)
    md = ndimage.median_filter(d, size=(1, 41))
    tops, bots = [], []
    for y in range(4, L.shape[0] - 4):
        for a, b in _runs(md[y] > thr, minlen):
            tops.append((y, a, b))
        for a, b in _runs(md[y] < -thr, minlen):
            bots.append((y, a, b))
    boxes = []
    for (yt, a, b) in tops:
        for (yb, c, e) in bots:
            if 30 <= yb - yt <= 450:
                inter = min(b, e) - max(a, c); union = max(b, e) - min(a, c)
                if inter > 0 and inter / union > 0.6:
                    boxes.append((yt, yb, min(a, c), max(b, e)))
    # boxes touching the frame edge: top edge only (box to bottom) or bottom only (box from top)
    for (yt, a, b) in tops:
        if yt > L.shape[0] - 460 and not any(bx[0] == yt for bx in boxes):
            inside = L[yt + 6:, a:b].mean(); above = L[max(yt - 40, 0):yt - 3, a:b].mean()
            if above - inside > 0.5:
                boxes.append((yt, L.shape[0], a, b))
    for (yb, c, e) in bots:
        if yb < 460 and not any(bx[1] == yb for bx in boxes):
            inside = L[:yb - 6, c:e].mean(); below = L[yb + 3:yb + 40, c:e].mean()
            if below - inside > 0.5:
                boxes.append((0, yb, c, e))
    return boxes


def subtitle_mask(x, margin=8):
    g = x.mean(-1)
    z = ndimage.binary_closing(x.max(-1) < 3, np.ones((25, 25)))
    bad = ndimage.binary_dilation(z, np.ones((9, 9)))
    blk = x.max(-1) < 3
    lab, n = ndimage.label(blk)
    for k, sl in enumerate(ndimage.find_objects(lab), 1):
        area = (lab[sl] == k).sum()
        if area > 3000 and area / blk[sl].size > 0.4:
            bad[max(sl[0].start - 9, 0):sl[0].stop + 9, max(sl[1].start - 9, 0):sl[1].stop + 9] = True
    for (y0, y1, x0, x1) in translucent_boxes(x):
        bad[max(y0 - margin, 0):y1 + margin, max(x0 - margin, 0):x1 + margin] = True
    # white neutral glyphs on a dark local background (text inside translucent boxes)
    th = g - ndimage.grey_opening(g, size=(9, 9))
    neutral = (x.max(-1) - x.min(-1)) < 16
    bg = ndimage.median_filter(g[::2, ::2], 15).repeat(2, 0).repeat(2, 1)[: g.shape[0], : g.shape[1]]
    glyph = (th > 50) & neutral & (g > 120) & (bg < 90)
    lines = ndimage.binary_closing(glyph, np.ones((13, 41)))
    lab, n = ndimage.label(lines)
    for sl in ndimage.find_objects(lab):
        h = sl[0].stop - sl[0].start; w = sl[1].stop - sl[1].start
        if 12 <= h <= 420 and w >= 80 and glyph[sl].mean() > 0.04:
            bad[max(sl[0].start - 30, 0):sl[0].stop + 30, max(sl[1].start - 30, 0):sl[1].stop + 30] = True
    sat = ndimage.binary_dilation(x.max(-1) > 250, np.ones((3, 3)))
    return bad | sat


