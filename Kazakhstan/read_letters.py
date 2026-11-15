"""Read each album's letter from fused candidate pools (absolute page frame).

Rewritten on 2026-11-15 after an audit of the first version:

* geometry and forensic evidence now live on one scale - a per-page mixture
  likelihood ``log(sum_i P_i * f(c_i) / u + P_0)`` - instead of a squared
  skeleton distance capped at 0.14 (max 0.0196) competing with
  ``0.22 * log(rel)`` (0.023 for a 10 % score drop), which froze the selection
  on the top-ranked candidate;
* template densities are normalised (``f`` integrates to one), so short
  glyphs cannot win by hiding inside a cluster, and candidates that fall off
  the template canvas receive the outlier density instead of the clipped
  border value;
* real albums and controls go through the *same* function
  (``score_pages``) with the same letters, fonts, models and transform grid;
* candidates of several detectors are fused by spatial clustering in page
  pixels instead of exact equality of rounded coordinates.

The reading frame is the absolute 1600x1200 page: the organisers' baseline
draws box centres at page coordinates.  No word list, guessed flag,
catalogue, duplicate stamp or manual coordinate is used.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
FONTS = ["arialbd.ttf", "arial.ttf", "verdanab.ttf", "timesbd.ttf",
         "cour.ttf", "consolab.ttf", "calibrib.ttf"]
MODELS = ["skeleton", "filled"]
H0, CANVAS = 200, 360              # glyph height and canvas (template pixels)
SIGMA = 0.032 * H0                 # stroke scatter, 3.2 % of the glyph height (~19 px)
SCALES = np.arange(400, 801, 40)   # glyph height on the page, pixels
CXS = np.arange(650, 951, 25)      # glyph-box centre on the page
CYS = np.arange(450, 726, 25)
P_MISS = 0.30                      # prior that no candidate of a page is true
STRONG_P = 0.90                    # prior of a single strong noise detection
MAX_PAGE_CONFIDENCE = 0.90         # cap on the summed candidate prior of a page


# ------------------------------------------------------------------ fusion
def fuse_candidates(sources, radius=48, absence=0.15, keep=8):
    """Cluster candidates of several detectors spatially (page pixels).

    ``sources``: {name: [(x, y, rel), ...]} for ONE page.  A candidate joins
    an existing cluster when it lies within ``radius`` horizontally and
    ``0.75 * radius`` vertically of the cluster centre (edit boxes are wider
    than tall; median 63 x 41 px), otherwise it starts a new cluster.  Each
    source contributes at most its best member per cluster; missing sources
    contribute ``absence``.  Distinct regions further apart are never merged.
    """
    items = sorted(((rel, x, y, name) for name, cands in sources.items()
                    for x, y, rel in cands), reverse=True)
    clusters = []
    for rel, x, y, name in items:
        for cl in clusters:
            if abs(cl["x"] - x) <= radius and abs(cl["y"] - y) <= 0.75 * radius:
                if name not in cl["rels"]:
                    cl["rels"][name] = rel
                    wsum = sum(cl["rels"].values())
                    cl["x"] += (x - cl["x"]) * rel / wsum
                    cl["y"] += (y - cl["y"]) * rel / wsum
                break
        else:
            clusters.append({"x": float(x), "y": float(y), "rels": {name: rel}})
    n = max(1, len(sources))
    out = [{"x": cl["x"], "y": cl["y"], "votes": len(cl["rels"]),
            "rel": (sum(cl["rels"].values()) + absence * (n - len(cl["rels"]))) / n}
           for cl in clusters]
    out.sort(key=lambda c: -c["rel"])
    return out[:keep]


def page_priors(fused, strong=None, gamma=3.0, p_miss=P_MISS, radius=48):
    """Candidate prior probabilities for one page (sum = 1 - p_miss).

    ``strong`` is an optional (x, y) of a single strong re-noised patch; it
    receives ``STRONG_P`` and the remaining candidates share the rest.
    """
    cands = [dict(c) for c in fused]
    if strong is not None:
        sx, sy = strong
        hit = [c for c in cands if abs(c["x"] - sx) <= radius and abs(c["y"] - sy) <= 0.75 * radius]
        if hit:
            hit[0]["strong"] = True
        else:
            cands.append({"x": float(sx), "y": float(sy), "rel": 1.0, "votes": 0, "strong": True})
    w = np.array([max(c["rel"], 1e-6) ** gamma for c in cands])
    is_strong = np.array([c.get("strong", False) for c in cands])
    if is_strong.any():
        rest = w * ~is_strong
        rest = rest / rest.sum() * (1 - STRONG_P) * 0.5 if rest.sum() > 0 else rest
        prob = np.where(is_strong, STRONG_P, rest)
    else:
        prob = w / w.sum() * (1 - p_miss)
    for c, p in zip(cands, prob):
        c["prior"] = float(p)
    return cands


# ------------------------------------------------------------------ glyphs
def zhang_suen(mask):
    img = mask.astype(np.uint8).copy()
    nbrs = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]
    changed = True
    while changed:
        changed = False
        for phase in (0, 1):
            pad = np.pad(img, 1)
            p2, p3, p4, p5, p6, p7, p8, p9 = [
                pad[1 + dy:pad.shape[0] - 1 + dy, 1 + dx:pad.shape[1] - 1 + dx]
                for dy, dx in nbrs]
            seq = [p2, p3, p4, p5, p6, p7, p8, p9, p2]
            trans = sum((seq[i] == 0) & (seq[i + 1] == 1) for i in range(8))
            nsum = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9
            base = (img == 1) & (nsum >= 2) & (nsum <= 6) & (trans == 1)
            if phase == 0:
                cond = base & ((p2 * p4 * p6) == 0) & ((p4 * p6 * p8) == 0)
            else:
                cond = base & ((p2 * p4 * p8) == 0) & ((p2 * p6 * p8) == 0)
            if cond.any():
                img[cond] = 0
                changed = True
    return img > 0


def glyph_density(letter, font_path, model, cache):
    """Normalised density map (sums to 1 over template pixels) and its mask."""
    key = (letter, str(font_path), model)
    if key in cache:
        return cache[key]
    import cv2
    font = ImageFont.truetype(str(font_path), 300)
    img = Image.new("L", (700, 700), 0)
    ImageDraw.Draw(img).text((150, 120), letter, fill=255, font=font)
    arr = np.asarray(img) > 128
    ys, xs = np.where(arr)
    crop = arr[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = crop.shape
    rw = max(1, int(round(w * H0 / h)))
    glyph = cv2.resize(crop.astype(np.uint8), (rw, H0), interpolation=cv2.INTER_AREA) > 0
    canvas = np.zeros((CANVAS, CANVAS), bool)
    oy, ox = (CANVAS - H0) // 2, (CANVAS - rw) // 2
    if rw > CANVAS:
        raise ValueError("glyph wider than canvas")
    canvas[oy:oy + H0, ox:ox + rw] = glyph
    target = zhang_suen(canvas) if model == "skeleton" else canvas
    dist = cv2.distanceTransform((~target).astype(np.uint8), cv2.DIST_L2, 5)
    dens = np.exp(-0.5 * (dist / SIGMA) ** 2).astype(np.float64)
    dens /= dens.sum()
    cache[key] = (dens, target)
    return cache[key]


# ------------------------------------------------------------------ scoring
def pack_pages(pages):
    """pages: list of candidate lists (dicts with x, y, prior, area)."""
    xs, ys, ps, idx, areas = [], [], [], [], []
    for i, page in enumerate(pages):
        for c in page:
            xs.append(c["x"]); ys.append(c["y"]); ps.append(c["prior"])
            idx.append(i); areas.append(c["area"])
    return (np.array(xs, float), np.array(ys, float), np.array(ps, float),
            np.array(idx, int), np.array(areas, float), len(pages))


def template_loglik(packed, dens, p_miss=None, scales=None, cys=None, cxs=None):
    """Log-likelihood ratio (vs. no-letter null) on the transform grid.

    Returns array [len(scales), len(cys), len(cxs)]; defaults to the coarse
    module grid.
    """
    scales = SCALES if scales is None else np.asarray(scales)
    cys = CYS if cys is None else np.asarray(cys)
    cxs = CXS if cxs is None else np.asarray(cxs)
    xs, ys, ps, idx, areas, n_pages = packed
    miss = np.zeros(n_pages)
    np.add.at(miss, idx, ps)
    miss = np.clip(1 - miss, 1e-3, 1) if p_miss is None else np.full(n_pages, p_miss)
    cx = cxs[None, :, None]
    cy = cys[:, None, None]
    out = np.zeros((len(scales), len(cys), len(cxs)))
    for si, s in enumerate(scales):
        k = H0 / s
        gx = np.rint((xs[None, None, :] - cx) * k + CANVAS / 2).astype(int)
        gy = np.rint((ys[None, None, :] - cy) * k + CANVAS / 2).astype(int)
        gx, gy = np.broadcast_arrays(gx, gy)
        ok = (gx >= 0) & (gx < CANVAS) & (gy >= 0) & (gy < CANVAS)
        vals = np.where(ok, dens[np.clip(gy, 0, CANVAS - 1), np.clip(gx, 0, CANVAS - 1)], 0.0)
        ratio = vals * (k * k) * areas[None, None, :] * ps[None, None, :]
        per_page = np.zeros(ratio.shape[:2] + (n_pages,))
        np.add.at(per_page, (slice(None), slice(None), idx), ratio)
        out[si] = np.log(per_page + miss[None, None, :]).sum(-1)
    return out


def score_pages(pages, fonts, cache, letters=LETTERS, models=MODELS,
                scales=None, cys=None, cxs=None):
    """Shared entry point for real albums AND controls.

    Returns {(font, model): array[len(letters), S, CY, CX]}.
    """
    packed = pack_pages(pages)
    table = {}
    for font in fonts:
        for model in models:
            grid = np.stack([template_loglik(packed, glyph_density(l, font, model, cache)[0],
                                             scales=scales, cys=cys, cxs=cxs)
                             for l in letters])
            table[(Path(font).name, model)] = grid
    return table


def summarise(table, letters=LETTERS):
    """Best letter ranking over fonts, models and transforms."""
    best = {}
    for (font, model), grid in table.items():
        flat = grid.reshape(len(letters), -1)
        arg = flat.argmax(1)
        for li, l in enumerate(letters):
            v = float(flat[li, arg[li]])
            if l not in best or v > best[l]["ll"]:
                si, yi, xi = np.unravel_index(arg[li], grid.shape[1:])
                best[l] = {"ll": v, "font": font, "model": model,
                           "scale": int(SCALES[si]), "cy": int(CYS[yi]), "cx": int(CXS[xi])}
    ranking = sorted(best.items(), key=lambda kv: -kv[1]["ll"])
    return ranking


def posterior_selection(pages, dens, scale, cx, cy):
    """Per-page posterior of each candidate under one fitted template."""
    k = H0 / scale
    chosen = []
    for page in pages:
        post = []
        for c in page:
            gx = int(round((c["x"] - cx) * k + CANVAS / 2))
            gy = int(round((c["y"] - cy) * k + CANVAS / 2))
            f = dens[gy, gx] if 0 <= gx < CANVAS and 0 <= gy < CANVAS else 0.0
            post.append(c["prior"] * f * k * k * c["area"])
        miss = max(1e-3, 1 - sum(c["prior"] for c in page))
        total = sum(post) + miss
        j = int(np.argmax(post)) if post else -1
        chosen.append({"index": j, "posterior": post[j] / total if post else 0.0,
                       "miss_posterior": miss / total})
    return chosen


# ------------------------------------------------------------------ data
def load_trufor(paths):
    out = {}
    for path in paths:
        name = Path(path).stem
        for r in csv.DictReader(open(path, newline="", encoding="utf-8")):
            out.setdefault((r["album"], r["page"]), {}).setdefault(name, []).append(
                (float(r["x"]), float(r["y"]), float(r["rel"])))
    return out


def load_noise(path):
    """Single-strong-stamp pages from noise_forensics.py -> {(album, page): (x, y)}."""
    if path is None:
        return {}
    return {(q["album"], q["page"]): (q["x"], q["y"]) for q in json.loads(Path(path).read_text())}


def build_pool(trufor, noise, areas, gamma=3.0, radius=48, use_noise=True):
    albums = {}
    for (album, page), sources in sorted(trufor.items()):
        fused = fuse_candidates(sources, radius=radius)
        cands = page_priors(fused, noise.get((album, page)) if use_noise else None,
                            gamma=gamma, radius=radius)
        for c in cands:
            c["area"] = areas[(album, page)]
        albums.setdefault(album, {})[page] = cands
    return albums


def load_rescored(path, min_prior=0.0, keep=8, strong_p=None, strong_pages=None):
    """rescore.py output -> {album: {page: [candidates with prior and area]}}.

    ``strong_p`` (optional) restores the noise detector's own precision on
    single-strong pages: the logistic rescoring is dominated by TruFor
    features and dilutes a strong re-noised patch (measured 80-90 % precise
    on single-strong pages) to a small prior whenever TruFor is weak there -
    exactly the family where TruFor is least reliable.  With ``strong_p``
    set (use STRONG_P), a page's strong candidate receives that prior, the
    page's other candidates share ``0.5 * (1 - strong_p)`` and the page's
    miss probability is ``0.5 * (1 - strong_p)``, as ``page_priors`` does in
    the unfused path.  ``strong_pages`` optionally restricts this to the
    shape-verified single-strong pages of noise_forensics.py (--pseudo JSON);
    the CSV ``strong`` column skips the shape check, so the restriction keeps
    the measured precision.  Pages without a strong candidate are untouched.
    """
    albums = {}
    for r in csv.DictReader(open(path, newline="", encoding="utf-8")):
        page = albums.setdefault(r["album"], {}).setdefault(r["page"], [])
        strong_flag = float(r.get("strong", 0)) > 0
        # a strong noise candidate is never dropped by the keep cap: the
        # logistic dilutes it precisely because TruFor is weak there
        if len(page) < keep or (strong_flag and strong_p is not None):
            if float(r["prior"]) >= min_prior or strong_flag:
                page.append({"x": float(r["x"]), "y": float(r["y"]), "prior": float(r["prior"]),
                             "area": float(r["area"]), "p": float(r["p"]),
                             "strong": strong_flag})
    # never be more than MAX_PAGE_CONFIDENCE sure about a page: 30 labelled
    # exhibits cannot certify a calibration tighter than that
    for album, pages in albums.items():
        for key, page in pages.items():
            full_key = (album, key)
            if strong_p is None or (strong_pages is not None and full_key not in strong_pages):
                total = sum(c["prior"] for c in page)
                if total > MAX_PAGE_CONFIDENCE:
                    for c in page:
                        c["prior"] *= MAX_PAGE_CONFIDENCE / total
                continue
            anchor = strong_pages.get(full_key) if strong_pages else None
            strong = [c for c in page if c["strong"]]
            lead = None
            if strong:
                lead = max(strong, key=lambda c: c["prior"])
            elif anchor is not None:
                near = [c for c in page
                        if abs(c["x"] - anchor[0]) <= 48 and abs(c["y"] - anchor[1]) <= 36]
                if near:
                    lead = max(near, key=lambda c: c["prior"])
                    lead["strong"] = True
                else:
                    lead = {"x": float(anchor[0]), "y": float(anchor[1]), "prior": 0.0,
                            "area": page[0]["area"] if page else 0.0, "p": 0.0,
                            "strong": True}
                    page.append(lead)
            if lead is None:
                total = sum(c["prior"] for c in page)
                if total > MAX_PAGE_CONFIDENCE:
                    for c in page:
                        c["prior"] *= MAX_PAGE_CONFIDENCE / total
                continue
            rest = [c for c in page if c is not lead]
            lead["prior"] = strong_p
            share = 0.5 * (1 - strong_p)
            for c in rest:
                c["prior"] = share / len(rest) if rest else 0.0
    return {a: dict(sorted(albums[a].items())) for a in sorted(albums)}


def write_outputs(albums, report, fonts, cache, output):
    """Selected boxes (one per page) and dot panels for the per-album best fit."""
    font_by_name = {Path(f).name: f for f in fonts}
    rows = []
    panel = Image.new("RGB", (5 * 400, 330), "white")
    draw = ImageDraw.Draw(panel)
    for ai, album in enumerate(sorted(albums)):
        best = report[album]["ranking"][0]
        dens, target = glyph_density(best["letter"], font_by_name[best["font"]], best["model"], cache)
        names = sorted(albums[album])
        pages = [albums[album][n] for n in names]
        chosen = posterior_selection(pages, dens, best["scale"], best["cx"], best["cy"])
        draw.rectangle((400 * ai + 1, 25, 400 * ai + 398, 325), outline="gray")
        draw.text((400 * ai + 8, 6), f"{album}: fit {best['letter']} "
                  f"(margin {report[album]['margin']:.1f})", fill="black")
        ys, xs = np.where(target)
        k = best["scale"] / H0
        for gx, gy in zip(xs[::7], ys[::7]):
            px = (gx - CANVAS / 2) * k + best["cx"]
            py = (gy - CANVAS / 2) * k + best["cy"]
            draw.point((400 * ai + px / 4, 25 + py / 4), fill=(200, 215, 255))
        for name, page, ch in zip(names, pages, chosen):
            if not page:
                continue
            fitted = page[ch["index"]]
            prior_top = max(page, key=lambda c: c["prior"])
            rows.append({"album": album, "page": name,
                         "x": int(round(prior_top["x"])) - 32, "y": int(round(prior_top["y"])) - 24,
                         "w": 64, "h": 48, "prior": round(prior_top["prior"], 4),
                         "fit_x": int(round(fitted["x"])) - 32, "fit_y": int(round(fitted["y"])) - 24,
                         "fit_posterior": round(float(ch["posterior"]), 4),
                         "fit_changed_choice": int(fitted is not prior_top)})
            x, y = 400 * ai + prior_top["x"] / 4, 25 + prior_top["y"] / 4
            r = 2 + 4 * prior_top["prior"]
            draw.ellipse((x - r, y - r, x + r, y + r), fill="black")
    with (output / "boxes.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    panel.save(output / "panels.png")


def stamp_areas(data, keys):
    import cv2
    from geometry import stamps
    out = {}
    for album, page in keys:
        im = cv2.imread(str(Path(data) / "albums" / album / f"{page}.jpg"))
        out[(album, page)] = float(sum((w - 12) * (h - 12) for _, _, w, h in stamps(im)))
    return out


def mixed_albums(albums, n_controls, rng, shuffle_priors=False):
    """Controls: 26 pages drawn across albums (or priors shuffled in-page)."""
    names = sorted(albums)
    flat = [albums[a][p] for a in names for p in sorted(albums[a])]
    owner = [a for a in names for _ in albums[a]]
    controls = []
    for _ in range(n_controls):
        if shuffle_priors:
            src = albums[names[rng.integers(len(names))]]
            pages = []
            for p in sorted(src):
                page = [dict(c) for c in src[p]]
                perm = rng.permutation(len(page))
                pri = [page[j]["prior"] for j in perm]
                for c, v in zip(page, pri):
                    c["prior"] = v
                pages.append(page)
        else:
            # stratified: at most ceil(26/len(names)) + 1 pages from one album
            order = rng.permutation(len(flat))
            cap, count, pages = 26 // len(names) + 2, {}, []
            for j in order:
                if count.get(owner[j], 0) < cap:
                    pages.append(flat[j]); count[owner[j]] = count.get(owner[j], 0) + 1
                if len(pages) == 26:
                    break
        controls.append(pages)
    return controls


def cross_page_albums(albums, n_controls, rng):
    """Controls: same pages, sizes and per-page priors, but candidate
    positions permuted across all pages of one album.

    This destroys the cross-page letter configuration (the only structure
    the fit reads) while keeping every page's candidate count, prior mass
    and stamp geometry identical to the real album.
    """
    controls = []
    for _ in range(n_controls):
        src = albums[sorted(albums)[int(rng.integers(len(albums)))]]
        names = sorted(src)
        coords = [(c["x"], c["y"]) for p in names for c in src[p]]
        perm = rng.permutation(len(coords))
        pages, i = [], 0
        for p in names:
            page = []
            for c in src[p]:
                x, y = coords[perm[i]]
                i += 1
                page.append({**c, "x": float(x), "y": float(y)})
            pages.append(page)
        controls.append(pages)
    return controls


def joint_fit(tables):
    """One (font, model, scale, cy, cx) shared by all albums."""
    best = None
    for key in next(iter(tables.values())):
        total = sum(tables[a][key].max(0) for a in tables)       # [S, CY, CX]
        j = np.unravel_index(total.argmax(), total.shape)
        if best is None or total[j] > best["ll"]:
            best = {"ll": float(total[j]), "font": key[0], "model": key[1],
                    "scale": int(SCALES[j[0]]), "cy": int(CYS[j[1]]), "cx": int(CXS[j[2]]),
                    "_key": key, "_j": tuple(int(v) for v in j)}
    key, j = best.pop("_key"), best.pop("_j")
    best["letters"] = {}
    for a in sorted(tables):
        col = tables[a][key][(slice(None),) + j]
        order = np.argsort(-col)
        best["letters"][a] = [{"letter": LETTERS[i], "ll": float(col[i])} for i in order[:5]]
    return best


def consistency(tables):
    """Summed best log-likelihood under three sharing regimes.

    independent : every album picks its own font, model and transform;
    partial     : font, model, scale and vertical centre shared, cx free;
    shared      : everything shared (``joint_fit``).
    The costs of sharing (independent - partial, independent - shared) and the
    per-member margins under the shared transform are the statistics compared
    against control album sets.
    """
    independent = sum(max(float(g.max()) for g in tables[a].values()) for a in tables)
    partial = max(float(sum(tables[a][key].max(0).max(-1) for a in tables).max())
                  for key in next(iter(tables.values())))
    shared_ll, shared_margins, shared_key, shared_j = None, None, None, None
    for key in next(iter(tables.values())):
        total = sum(tables[a][key].max(0) for a in tables)
        j = np.unravel_index(total.argmax(), total.shape)
        if shared_ll is None or total[j] > shared_ll:
            shared_ll, shared_key, shared_j = float(total[j]), key, j
    margins = []
    for a in sorted(tables):
        col = tables[a][shared_key][(slice(None),) + shared_j]
        top2 = np.sort(col)[-2:]
        margins.append(float(top2[1] - top2[0]))
    return {"independent": independent, "partial": partial, "shared": shared_ll,
            "cost_partial": independent - partial, "cost_shared": independent - shared_ll,
            "shared_margins": margins,
            "shared_center": {"font": shared_key[0], "model": shared_key[1],
                              "scale": int(SCALES[shared_j[0]]), "cy": int(CYS[shared_j[1]]),
                              "cx": int(CXS[shared_j[2]])}}


def fine_grids(scale, cy, cx):
    """Fine transform grid around a coarse optimum (coarse steps 40/25/25 px
    sit between the true glyph-box parameters; at the true box the true
    letter's points align on the glyph strokes and gain more than rivals)."""
    fs = np.arange(max(int(SCALES[0]), scale - 40), min(int(SCALES[-1]), scale + 40) + 1, 10)
    fcy = np.arange(max(int(CYS[0]), cy - 24), min(int(CYS[-1]), cy + 24) + 1, 12)
    fcx = np.arange(max(int(CXS[0]), cx - 24), min(int(CXS[-1]), cx + 24) + 1, 12)
    return fs, fcy, fcx


def fine_stage(pages_by_key, center, fonts_by_name, cache):
    """Second stage: re-score every album on a fine grid around the shared
    coarse optimum, for the joint (font, model) only.  Returns per-album
    letter rankings, margins at the fine shared optimum and the fine cost of
    sharing.  Real albums and every control group go through this identical
    function.
    """
    font, model = center["font"], center["model"]
    fs, fcy, fcx = fine_grids(center["scale"], center["cy"], center["cx"])
    ftables = {}
    for key, pages in pages_by_key.items():
        packed = pack_pages(pages)
        ftables[key] = np.stack(
            [template_loglik(packed, glyph_density(l, fonts_by_name[font], model, cache)[0],
                             scales=fs, cys=fcy, cxs=fcx) for l in LETTERS])
    total = sum(ftables[k].max(0) for k in ftables)
    j = np.unravel_index(total.argmax(), total.shape)
    out = {"center": {"font": font, "model": model, "scale": int(fs[j[0]]),
                      "cy": int(fcy[j[1]]), "cx": int(fcx[j[2]])}}
    margins, letters = {}, {}
    for k in sorted(ftables):
        col = ftables[k][(slice(None),) + j]
        order = np.argsort(-col)
        letters[k] = [{"letter": LETTERS[i], "ll": float(col[i])} for i in order[:5]]
        margins[k] = float(col[order[0]] - col[order[1]])
    out["letters"] = letters
    out["margins"] = margins
    out["shared_ll"] = float(total[j])
    out["independent_ll"] = sum(float(ftables[k].max()) for k in ftables)
    out["cost_shared"] = out["independent_ll"] - out["shared_ll"]
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, nargs="+", default=[],
                        help="TruFor candidate CSVs (candidates.py)")
    parser.add_argument("--rescored", type=Path,
                        help="rescore.py CSV with calibrated priors (preferred)")
    parser.add_argument("--noise", type=Path, help="noise_forensics.py --pseudo JSON")
    parser.add_argument("--data", type=Path, help="dataset root (only for --candidates)")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fonts", type=Path, default=Path("C:/Windows/Fonts"))
    parser.add_argument("--gamma", type=float, default=3.0)
    parser.add_argument("--radius", type=float, default=48)
    parser.add_argument("--controls", type=int, default=20)
    parser.add_argument("--keep", type=int, default=8,
                        help="candidates kept per page from the rescored CSV")
    parser.add_argument("--strong-priors", action="store_true",
                        help="give a page's strong noise candidate the "
                             "detector's own precision prior (STRONG_P)")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--no-noise", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    fonts = [args.fonts / f for f in FONTS if (args.fonts / f).exists()]
    if args.rescored:
        strong_pages = None
        if args.strong_priors and args.noise:
            strong_pages = load_noise(args.noise)
        albums = load_rescored(args.rescored, keep=args.keep,
                               strong_p=STRONG_P if args.strong_priors else None,
                               strong_pages=strong_pages)
    else:
        trufor = load_trufor(args.candidates)
        noise = load_noise(args.noise)
        areas = stamp_areas(args.data, trufor.keys())
        albums = build_pool(trufor, noise, areas, args.gamma, args.radius, not args.no_noise)
    cache, report, tables = {}, {}, {}
    for album in sorted(albums):
        pages = [albums[album][p] for p in sorted(albums[album])]
        tables[album] = score_pages(pages, fonts, cache)
        ranking = summarise(tables[album])
        report[album] = {"ranking": [{"letter": l, **v} for l, v in ranking[:8]],
                         "margin": ranking[0][1]["ll"] - ranking[1][1]["ll"]}
        print(album, " ".join(f"{l}:{v['ll']:.1f}" for l, v in ranking[:6]),
              "| best", ranking[0][1], flush=True)
    # cross-album consistency: independent vs. partially vs. fully shared
    report["joint"] = joint_fit(tables)
    report["consistency"] = consistency(tables)
    print("joint", {k: v for k, v in report["joint"].items() if k != "letters"})
    for a, top in report["joint"]["letters"].items():
        print("  ", a, " ".join(f"{t['letter']}:{t['ll']:.1f}" for t in top))
    print("consistency", report["consistency"], flush=True)
    # controls through the identical pipeline
    for name, shuffle in (("mixed", False), ("prior_shuffled", True)):
        best, margins = [], []
        for pages in mixed_albums(albums, args.controls, rng, shuffle):
            ranking = summarise(score_pages(pages, fonts, cache))
            best.append(ranking[0][1]["ll"])
            margins.append(ranking[0][1]["ll"] - ranking[1][1]["ll"])
        report[f"control_{name}"] = {"best_ll": best, "margin": margins}
        print(f"control {name}: best LL mean {np.mean(best):.1f} max {np.max(best):.1f}; "
              f"margin mean {np.mean(margins):.1f} max {np.max(margins):.1f}", flush=True)
    best, margins = [], []
    for pages in cross_page_albums(albums, args.controls, rng):
        ranking = summarise(score_pages(pages, fonts, cache))
        best.append(ranking[0][1]["ll"])
        margins.append(ranking[0][1]["ll"] - ranking[1][1]["ll"])
    report["control_cross_page"] = {"best_ll": best, "margin": margins}
    print(f"control cross_page: best LL mean {np.mean(best):.1f} max {np.max(best):.1f}; "
          f"margin mean {np.mean(margins):.1f} max {np.max(margins):.1f}", flush=True)
    costs = []
    fine_controls = []
    fonts_by_name = {Path(f).name: f for f in fonts}
    for _ in range(max(1, args.controls // 5)):
        group = mixed_albums(albums, len(albums), rng)
        ctables = {i: score_pages(pages, fonts, cache) for i, pages in enumerate(group)}
        cc = consistency(ctables)
        costs.append(cc)
        fc = fine_stage({i: pages for i, pages in enumerate(group)},
                        cc["shared_center"], fonts_by_name, cache)
        fine_controls.append({"margins": [fc["margins"][i] for i in sorted(fc["margins"])],
                              "cost_shared": fc["cost_shared"],
                              "letters": [fc["letters"][i][0]["letter"]
                                          for i in sorted(fc["letters"])],
                              "center": fc["center"]})
    report["control_consistency"] = costs
    report["control_fine"] = fine_controls
    print("control consistency cost_shared:",
          [round(c["cost_shared"], 1) for c in costs], "cost_partial:",
          [round(c["cost_partial"], 1) for c in costs], flush=True)
    # fine refinement of the real albums around the coarse joint optimum
    center = {"font": report["joint"]["font"], "model": report["joint"]["model"],
              "scale": report["joint"]["scale"], "cy": report["joint"]["cy"],
              "cx": report["joint"]["cx"]}
    fine = fine_stage({a: [albums[a][p] for p in sorted(albums[a])] for a in albums},
                      center, fonts_by_name, cache)
    report["fine"] = fine
    null_margins = [m for fc in fine_controls for m in fc["margins"]]
    report["fine"]["null_margin_p95"] = float(np.percentile(null_margins, 95))
    report["fine"]["null_margin_max"] = float(np.max(null_margins))
    print("fine center", fine["center"], "margins",
          {k: round(v, 2) for k, v in fine["margins"].items()},
          "letters", {k: fine["letters"][k][0]["letter"] for k in fine["letters"]},
          f"| null p95 {report['fine']['null_margin_p95']:.2f} "
          f"max {report['fine']['null_margin_max']:.2f} "
          f"cost_shared {fine['cost_shared']:.1f} vs controls "
          f"{[round(fc['cost_shared'], 1) for fc in fine_controls]}", flush=True)
    for album in sorted(albums):
        ll = report[album]["ranking"][0]["ll"]
        report[album]["p_mixed"] = float((np.array(report["control_mixed"]["best_ll"]) >= ll).mean())
    (args.output / "letters_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_outputs(albums, report, fonts, cache, args.output)
    pool = {a: {p: albums[a][p] for p in sorted(albums[a])} for a in albums}
    (args.output / "pool.json").write_text(json.dumps(pool), encoding="utf-8")


if __name__ == "__main__":
    main()
