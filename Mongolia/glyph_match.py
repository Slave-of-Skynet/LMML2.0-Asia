"""Objective glyph decoding of the 9x37 count matrix.

Hypothesis tested: 6 glyphs of 5x7 cells, 1-cell gaps, 1-cell border (6*6+1 = 37 cols, 7+2 = 9 rows).
Each 5x7 block (z-scored counts) is correlated with every letter of a standard 5x7 A-Z font; prints
top-3 letters with Pearson r per block for each orientation, plus the background/gap statistics.

  python glyph_match.py --matrix grids/<tag>_matrix.csv
"""
import argparse

import numpy as np
import pandas as pd

FONT = {
    "A": [".XXX.", "X...X", "X...X", "XXXXX", "X...X", "X...X", "X...X"],
    "B": ["XXXX.", "X...X", "X...X", "XXXX.", "X...X", "X...X", "XXXX."],
    "C": [".XXX.", "X...X", "X....", "X....", "X....", "X...X", ".XXX."],
    "D": ["XXXX.", "X...X", "X...X", "X...X", "X...X", "X...X", "XXXX."],
    "E": ["XXXXX", "X....", "X....", "XXXX.", "X....", "X....", "XXXXX"],
    "F": ["XXXXX", "X....", "X....", "XXXX.", "X....", "X....", "X...."],
    "G": [".XXX.", "X...X", "X....", "X.XXX", "X...X", "X...X", ".XXXX"],
    "H": ["X...X", "X...X", "X...X", "XXXXX", "X...X", "X...X", "X...X"],
    "I": [".XXX.", "..X..", "..X..", "..X..", "..X..", "..X..", ".XXX."],
    "J": ["..XXX", "...X.", "...X.", "...X.", "...X.", "X..X.", ".XX.."],
    "K": ["X...X", "X..X.", "X.X..", "XX...", "X.X..", "X..X.", "X...X"],
    "L": ["X....", "X....", "X....", "X....", "X....", "X....", "XXXXX"],
    "M": ["X...X", "XX.XX", "X.X.X", "X.X.X", "X...X", "X...X", "X...X"],
    "N": ["X...X", "X...X", "XX..X", "X.X.X", "X..XX", "X...X", "X...X"],
    "O": [".XXX.", "X...X", "X...X", "X...X", "X...X", "X...X", ".XXX."],
    "P": ["XXXX.", "X...X", "X...X", "XXXX.", "X....", "X....", "X...."],
    "Q": [".XXX.", "X...X", "X...X", "X...X", "X.X.X", "X..X.", ".XX.X"],
    "R": ["XXXX.", "X...X", "X...X", "XXXX.", "X.X..", "X..X.", "X...X"],
    "S": [".XXXX", "X....", "X....", ".XXX.", "....X", "....X", "XXXX."],
    "T": ["XXXXX", "..X..", "..X..", "..X..", "..X..", "..X..", "..X.."],
    "U": ["X...X", "X...X", "X...X", "X...X", "X...X", "X...X", ".XXX."],
    "V": ["X...X", "X...X", "X...X", "X...X", "X...X", ".X.X.", "..X.."],
    "W": ["X...X", "X...X", "X...X", "X.X.X", "X.X.X", "X.X.X", ".X.X."],
    "X": ["X...X", "X...X", ".X.X.", "..X..", ".X.X.", "X...X", "X...X"],
    "Y": ["X...X", "X...X", ".X.X.", "..X..", "..X..", "..X..", "..X.."],
    "Z": ["XXXXX", "....X", "...X.", "..X..", ".X...", "X....", "XXXXX"],
}
T = {k: np.array([[c == "X" for c in r] for r in v], float) for k, v in FONT.items()}


def blocks(M):
    return [M[1:8, 1 + 6 * k:6 + 6 * k] for k in range(6)]


def corr(a, b):
    a = a.ravel() - a.mean()
    b = b.ravel() - b.mean()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def decode(M):
    out = []
    for B in blocks(M):
        s = sorted(((corr(B, t), k) for k, t in T.items()), reverse=True)
        out.append(s[:3])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", required=True)
    a = ap.parse_args()
    M = pd.read_csv(a.matrix).to_numpy(float)
    assert M.shape == (9, 37), M.shape
    gap = np.zeros_like(M, bool)
    gap[[0, 8], :] = True
    gap[:, 0::6] = True
    print(f"border/gap cells: n={gap.sum()} mean={M[gap].mean():.1f} sd={M[gap].std():.1f} | "
          f"glyph-area cells: n={(~gap).sum()} mean={M[~gap].mean():.1f} sd={M[~gap].std():.1f}")
    for name, A in [("original", M), ("flip_lr", M[:, ::-1]), ("flip_ud", M[::-1]), ("rot180", M[::-1, ::-1])]:
        d = decode(A)
        word = "".join(x[0][1] for x in d)
        mean_r = np.mean([x[0][0] for x in d])
        detail = "  ".join(f"{x[0][1]}{x[0][0]:.2f}({x[1][1]}{x[1][0]:.2f})" for x in d)
        print(f"{name:9s} {word}  mean_r={mean_r:.3f}  {detail}")
    best = decode(M)
    on_mask = np.zeros_like(M, bool)
    for k, x in enumerate(best):
        on_mask[1:8, 1 + 6 * k:6 + 6 * k] = T[x[0][1]] > 0
    print(f"original orientation, best-letter mask: ON cells n={on_mask.sum()} mean={M[on_mask].mean():.1f} "
          f"| OFF cells n={(~on_mask).sum()} mean={M[~on_mask].mean():.1f} sd={M[~on_mask].std():.1f}")


if __name__ == "__main__":
    main()
