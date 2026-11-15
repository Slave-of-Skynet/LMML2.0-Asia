"""Noise residual extraction + cache (exploration helper)."""
import os, sys, glob
import numpy as np
import pywt
from PIL import Image
from scipy import ndimage
from concurrent.futures import ProcessPoolExecutor

DATA = os.environ.get("JAPAN_DATASET", "dataset")
CACHE = os.environ.get("JAPAN_CACHE", "cache")


def load(path):
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)


def wavelet_denoise_residual(x, sigma0=3.0, levels=4, wave="db8"):
    """Mihcak-style local Wiener in wavelet domain; returns noise residual x - denoised."""
    coeffs = pywt.wavedec2(x, wave, level=levels, mode="periodization")
    out = [coeffs[0]]
    s2 = sigma0 ** 2
    for det in coeffs[1:]:
        nd = []
        for d in det:
            best = None
            for w in (3, 5, 7, 9):
                v = np.maximum(ndimage.uniform_filter(d * d, w) - s2, 0)
                best = v if best is None else np.minimum(best, v)
            nd.append(d * best / (best + s2))
        out.append(tuple(nd))
    den = pywt.waverec2(out, wave, mode="periodization")[: x.shape[0], : x.shape[1]]
    return x - den


def zero_mean(w):
    w = w - w.mean(axis=0, keepdims=True)
    w = w - w.mean(axis=1, keepdims=True)
    return w


def process(path):
    name = os.path.relpath(path, DATA).replace(os.sep, "__")
    out = os.path.join(CACHE, name + ".npz")
    if os.path.exists(out):
        return out
    rgb = load(path)
    res = np.stack([wavelet_denoise_residual(rgb[..., c]) for c in range(3)], -1)
    np.savez(out, res=res.astype(np.float16))
    return out


