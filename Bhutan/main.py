"""Reassemble vertical PNG strips using directed boundary compatibility.

Inference needs only this file and input/*.png beside it; no reference image.
"""
import os
import time

START = time.perf_counter()
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_key] = "1"

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

cv2.setNumThreads(1)


def load_strips(directory):
    paths = sorted(p for p in Path(directory).iterdir()
                   if p.is_file() and p.suffix.lower() == ".png")
    if not paths:
        raise ValueError(f"No PNG strips in {directory}")
    images = [cv2.imdecode(np.fromfile(p, np.uint8), cv2.IMREAD_UNCHANGED)
              for p in paths]
    if any(im is None for im in images):
        raise ValueError("An input PNG could not be decoded")
    if any(im.shape != images[0].shape or im.dtype != images[0].dtype
           for im in images):
        raise ValueError("Strips must have identical dimensions and dtype")
    return paths, images


def boundary_features(images, max_height=1024):
    h = images[0].shape[0]
    rows = np.linspace(0, h - 1, min(h, max_height)).astype(int)
    features = []
    for image in images:
        x = image[rows].astype(np.float64)
        if image.dtype == np.uint16:
            x *= 255.0 / 65535.0
        if x.ndim == 2:
            x = np.repeat(x[:, :, None], 3, axis=2)
        elif x.shape[2] == 4:
            x = x[:, :, :3] * (x[:, :, 3:] / 255.0)
        features.append(x[:, [0, min(1, x.shape[1]-1),
                               max(0, x.shape[1]-2), -1], :3])
    x = np.stack(features)
    return x[:, :, 0], x[:, :, 1], x[:, :, 2], x[:, :, 3]


def mahalanobis_side(predicted, target, gradient):
    """All pairs, without allocating an N x N x height x RGB tensor."""
    n, h, _ = predicted.shape
    centered = gradient - gradient.mean(axis=1, keepdims=True)
    covariance = np.einsum('nhc,nhd->ncd', centered, centered) / max(h-1, 1)
    ridge = np.maximum(1.0, np.trace(covariance, axis1=1, axis2=2) / 300)
    covariance += ridge[:, None, None] * np.eye(3)
    inverse = np.linalg.inv(covariance)
    weighted = np.einsum('nhc,ncd->nhd', predicted, inverse)
    self_term = np.sum(weighted * predicted, axis=(1, 2))
    moments = np.einsum('nhc,nhd->ncd', target, target)
    cost = self_term[:, None] + np.einsum('icd,jcd->ij', inverse, moments)
    cost -= 2 * (weighted.reshape(n, -1) @ target.reshape(n, -1).T)
    return np.maximum(cost / h, 0)


def boundary_costs(images, metric="hybrid"):
    left, next_left, prev_right, right = boundary_features(images)
    n, h, _ = left.shape
    lf, rf = left.reshape(n, -1), right.reshape(n, -1)
    if metric == "color":
        cost = cdist(rf, lf, "cityblock") / (h * 3)
    elif metric == "gradient":
        cost = (cdist((2*right-prev_right).reshape(n, -1), lf, "sqeuclidean")
                + cdist(rf, (2*left-next_left).reshape(n, -1), "sqeuclidean")) / (h*6)
    else:
        gr, gl = right-prev_right, next_left-left
        cost = (mahalanobis_side(right+gr.mean(axis=1, keepdims=True), left, gr)
                + mahalanobis_side(left-gl.mean(axis=1, keepdims=True), right, gl).T)
        if metric == "hybrid" and n > 1:
            # Absolute colour anchors low-variance regions where covariance
            # alone can make a distant smooth strip look deceptively cheap.
            color = cdist(rf, lf, "cityblock") / (h*3)
            mask = ~np.eye(n, dtype=bool)
            cost = (cost / max(float(np.median(cost[mask])), 1e-9)
                    + 0.1*color / max(float(np.median(color[mask])), 1e-9))
    np.fill_diagonal(cost, np.inf)
    return cost


def assignment_path(cost):
    """Minimum assignment then merge subtours, using a free dummy endpoint."""
    n = len(cost)
    augmented = np.zeros((n+1, n+1))
    augmented[:n, :n] = cost
    np.fill_diagonal(augmented, np.inf)
    rows, cols = linear_sum_assignment(augmented)
    successor = np.empty(n+1, dtype=int)
    successor[rows] = cols
    while True:
        labels = np.full(n+1, -1, dtype=int)
        count = 0
        for start in range(n+1):
            if labels[start] >= 0:
                continue
            cur = start
            while labels[cur] < 0:
                labels[cur] = count
                cur = successor[cur]
            count += 1
        if count == 1:
            break
        old = augmented[np.arange(n+1), successor]
        delta = (augmented[:, successor] + augmented[:, successor].T
                 - old[:, None] - old[None, :])
        delta[labels[:, None] == labels[None, :]] = np.inf
        a, c = np.unravel_index(np.argmin(delta), delta.shape)
        successor[a], successor[c] = successor[c], successor[a]
    path = []
    cur = successor[n]
    while cur != n:
        path.append(int(cur))
        cur = successor[cur]
    return path


def greedy_path(cost):
    n = len(cost)
    successor = np.full(n, -1, int)
    predecessor = np.full(n, -1, int)
    parent = list(range(n))

    def root(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    joined = 0
    for flat in np.argsort(cost, axis=None):
        a, b = divmod(int(flat), n)
        if a == b or successor[a] >= 0 or predecessor[b] >= 0:
            continue
        ra, rb = root(a), root(b)
        if ra == rb:
            continue
        successor[a], predecessor[b], parent[ra] = b, a, rb
        joined += 1
        if joined == n-1:
            break
    cur = int(np.flatnonzero(predecessor < 0)[0])
    path = []
    while cur >= 0:
        path.append(cur)
        cur = int(successor[cur])
    return path


def path_cost(cost, path):
    return float(np.sum(cost[np.asarray(path[:-1]), np.asarray(path[1:])]))


def improve(cost, path, deadline):
    """Bounded block relocation and directed 2-opt, with free endpoints."""
    n = len(path)
    c = np.zeros((n+1, n+1))
    c[:n, :n] = cost
    p = np.asarray(path, int)
    for _ in range(80):
        if time.perf_counter() >= deadline or n < 3:
            break
        best_delta, operation = -1e-8, None
        for k in range(1, min(4, n-1)+1):
            starts = np.arange(n-k+1)
            prev = np.r_[n, p[:-k]]
            after = np.r_[p[k:], n]
            first, last = p[:n-k+1], p[k-1:]
            gap_l, gap_r = np.r_[n, p], np.r_[p, n]
            delta = (c[prev, after]-c[prev, first]-c[last, after])[:, None]
            delta = delta + c[gap_l[None, :], first[:, None]]
            delta = delta + c[last[:, None], gap_r[None, :]]
            delta -= c[gap_l, gap_r][None, :]
            gaps = np.arange(n+1)
            delta[(gaps[None, :] >= starts[:, None]) &
                  (gaps[None, :] <= starts[:, None]+k)] = np.inf
            i, j = np.unravel_index(np.argmin(delta), delta.shape)
            if delta[i, j] < best_delta:
                best_delta, operation = float(delta[i, j]), ("move", i, j, k)
        prev, after = np.r_[n, p[:-1]], np.r_[p[1:], n]
        reverse_diff = c[p[1:], p[:-1]] - c[p[:-1], p[1:]]
        prefix = np.r_[0, np.cumsum(reverse_diff)]
        delta = (c[prev[:, None], p[None, :]] + c[p[:, None], after[None, :]]
                 - c[prev, p][:, None] - c[p, after][None, :]
                 + prefix[None, :] - prefix[:, None])
        delta[np.tril_indices(n)] = np.inf
        i, j = np.unravel_index(np.argmin(delta), delta.shape)
        if delta[i, j] < best_delta:
            operation = ("reverse", i, j, 0)
        if operation is None:
            break
        kind, i, j, k = operation
        if kind == "reverse":
            candidate = p.copy()
            candidate[i:j+1] = candidate[i:j+1][::-1]
        else:
            block = p[i:i+k]
            rest = np.r_[p[:i], p[i+k:]]
            insert = j-k if j > i else j
            candidate = np.r_[rest[:insert], block, rest[insert:]]
        if path_cost(cost, candidate) >= path_cost(cost, p)-1e-8:
            break
        p = candidate
    return p.tolist()


def reconstruct(images, metric="hybrid", deadline=None):
    if len(images) == 1:
        return [0], np.array([[np.inf]])
    if deadline is None:
        deadline = time.perf_counter()+12
    cost = boundary_costs(images, metric)
    candidates = [assignment_path(cost), greedy_path(cost)]
    candidates = [improve(cost, p, deadline) for p in candidates]
    path = min(candidates, key=lambda p: path_cost(cost, p))
    if sorted(path) != list(range(len(images))):
        raise RuntimeError("Reconstruction is not a permutation")
    return path, cost


def main():
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=base/"input")
    parser.add_argument("--output-dir", type=Path, default=base)
    parser.add_argument("--metric", choices=("color", "gradient", "mgc", "hybrid"), default="hybrid")
    args = parser.parse_args()
    paths, images = load_strips(args.input)
    order, _ = reconstruct(images, args.metric, START+15)
    result = np.concatenate([images[i] for i in order], axis=1)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".png", result, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    if not ok:
        raise OSError("PNG encoding failed")
    (args.output_dir/"reconstructed_image.png").write_bytes(encoded.tobytes())
    (args.output_dir/"reconstructed_order.json").write_text(
        json.dumps([paths[i].name for i in order], indent=2)+"\n", encoding="utf-8")
    print(json.dumps({"strips":len(order), "seconds":round(time.perf_counter()-START, 4)}))


if __name__ == "__main__":
    main()
