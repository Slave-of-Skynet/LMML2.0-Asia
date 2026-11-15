"""Reconstruct the 9x37 survey grid from survey_plots.csv and render a counts file onto it.

Usage:
  python grid_decode.py --counts counts.csv --plots data/survey_plots.csv --out outdir [--tag name]

Outputs (in outdir): <tag>_matrix.csv (9x37 counts, row 0 = top after rotation),
<tag>_grid.png (raw / normalized / binary in 4 orientations), grid geometry printout.
"""
import argparse
import os

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw


def build_grid(plots, pitch=50.0):
    xy = plots[["x_m", "y_m"]].to_numpy(float)
    c = xy - xy.mean(0)
    _, _, vt = np.linalg.svd(c, full_matrices=False)
    u = vt[0]
    if u[0] < 0:
        u = -u
    v = np.array([-u[1], u[0]])
    a = c @ u
    b = c @ v
    angle = np.degrees(np.arctan2(u[1], u[0]))
    pitch = float(pitch)
    pitch_b = pitch
    col = np.round((a - a.min()) / pitch).astype(int)
    row = np.round((b.max() - b) / pitch).astype(int)
    resid_a = (a - a.min()) / pitch - col
    resid_b = (b.max() - b) / pitch - row
    return dict(angle=angle, pitch=pitch, pitch_b=float(pitch_b), col=col, row=row,
                resid=float(np.max(np.abs(np.r_[resid_a, resid_b]))), a=a, b=b)


def matrix_from(counts, plots, g):
    m = plots[["image"]].copy()
    m["row"] = g["row"]
    m["col"] = g["col"]
    m = m.merge(counts, on="image", how="left")
    nr, nc = m.row.max() + 1, m.col.max() + 1
    M = np.full((nr, nc), np.nan)
    for r, cc, v in m[["row", "col", "count"]].itertuples(index=False):
        M[r, cc] = v
    return M, m


def _panel(A, lo, hi, cell=18, cmap="gray"):
    B = np.clip((A - lo) / max(hi - lo, 1e-9), 0, 1)
    B = np.nan_to_num(B, nan=0.0)
    if cmap == "gray":
        rgb = np.stack([B * 255] * 3, -1)
    else:
        rgb = np.stack([255 * np.clip(1.5 * B - 0.2, 0, 1), 255 * np.clip(2 * B - 0.6, 0, 1) * 0 + 255 * B ** 2, 255 * (1 - B) * 0.6], -1)
    img = Image.fromarray(rgb.astype(np.uint8)).resize((A.shape[1] * cell, A.shape[0] * cell), Image.NEAREST)
    return img


def render(M, out_png, title, cell=18):
    orients = [("original", M), ("flip_lr", M[:, ::-1]), ("flip_ud", M[::-1, :]), ("rot180", M[::-1, ::-1])]
    lo, hi = np.nanpercentile(M, 5), np.nanpercentile(M, 95)
    thr = (np.nanpercentile(M, 25) + np.nanpercentile(M, 90)) / 2
    w = M.shape[1] * cell
    h = M.shape[0] * cell
    canvas = Image.new("RGB", (3 * (w + 20) + 20, len(orients) * (h + 30) + 30), (255, 255, 255))
    d = ImageDraw.Draw(canvas)
    d.text((10, 5), title, fill=(0, 0, 0))
    for i, (name, A) in enumerate(orients):
        y = 30 + i * (h + 30)
        panels = [(_panel(A, lo, hi, cell, "heat"), f"{name} raw 5-95%"),
                  (_panel(A, lo, hi, cell), f"{name} gray"),
                  (_panel((A > thr).astype(float), 0, 1, cell), f"{name} > {thr:.1f}")]
        for j, (p, t) in enumerate(panels):
            x = 20 + j * (w + 20)
            canvas.paste(p, (x, y))
            d.text((x, y - 13), t, fill=(0, 0, 0))
            for k in range(0, A.shape[1], 6):
                d.rectangle([x + k * cell, y, x + (k + 1) * cell - 1, y + h - 1], outline=(255, 0, 0))
    canvas.save(out_png)

def render_scatter(plots, counts, out_png, scale=0.6):
    """survey_map.png: plots at (x_m, y_m), equal aspect (1 px = 1/scale m), colour = count, y up."""
    d = plots.merge(counts, on="image")
    x0, x1 = d.x_m.min() - 60, d.x_m.max() + 60
    y0, y1 = d.y_m.min() - 60, d.y_m.max() + 60
    W, H = int((x1 - x0) * scale), int((y1 - y0) * scale)
    img = Image.new("RGB", (W + 90, H + 40), (255, 255, 255))
    dr = ImageDraw.Draw(img)
    lo, hi = np.percentile(d["count"], 2), np.percentile(d["count"], 98)

    def col(v):
        t = float(np.clip((v - lo) / max(hi - lo, 1e-9), 0, 1))
        return (int(255 * min(1, 2 * t)), int(255 * t * t), int(160 * (1 - t) + 40))

    for x, y, v in d[["x_m", "y_m", "count"]].itertuples(index=False):
        px, py = (x - x0) * scale, (y1 - y) * scale + 20
        dr.ellipse([px - 11, py - 11, px + 11, py + 11], fill=col(v), outline=(0, 0, 0))
    for i in range(101):
        v = lo + (hi - lo) * i / 100
        y = 20 + H - H * i / 100
        dr.line([W + 20, y, W + 40, y], fill=col(v), width=max(1, H // 100 + 1))
    dr.text((W + 45, 15), f"{hi:.0f}", fill=(0, 0, 0))
    dr.text((W + 45, H + 15), f"{lo:.0f}", fill=(0, 0, 0))
    dr.text((10, 3), "Predicted sheep count per survey plot (x_m right, y_m up, equal aspect)", fill=(0, 0, 0))
    img.save(out_png)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--counts", required=True)
    ap.add_argument("--plots", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default="grid")
    ap.add_argument("--col", default="count")
    ap.add_argument("--pitch", type=float, default=50.0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    plots = pd.read_csv(args.plots)
    counts = pd.read_csv(args.counts)[["image", args.col]].rename(columns={args.col: "count"})
    assert set(counts.image) == set(plots.image), "image sets differ"
    g = build_grid(plots, args.pitch)
    M, m = matrix_from(counts, plots, g)
    cells = len(set(zip(m.row, m.col)))
    print(f"angle={g['angle']:.2f} deg pitch={g['pitch']:.2f} m grid={M.shape[0]}x{M.shape[1]} "
          f"unique_cells={cells}/{len(plots)} max_snap_resid={g['resid']:.3f} pitch")
    pd.DataFrame(M.astype(float)).to_csv(os.path.join(args.out, f"{args.tag}_matrix.csv"), index=False)
    m.to_csv(os.path.join(args.out, f"{args.tag}_cells.csv"), index=False)
    render(M, os.path.join(args.out, f"{args.tag}_grid.png"), f"{args.tag}: {args.counts}")
    render_scatter(plots, counts, os.path.join(args.out, f"{args.tag}_survey_map.png"))
    np.set_printoptions(linewidth=250)
    for r in range(M.shape[0]):
        print(" ".join(f"{int(v):3d}" if not np.isnan(v) else "  ." for v in M[r]))


if __name__ == "__main__":
    main()
