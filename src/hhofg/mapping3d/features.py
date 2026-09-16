from __future__ import annotations

import hashlib
import re

import cv2
import numpy as np


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    norm = float(np.linalg.norm(arr))
    if norm <= eps:
        return np.zeros_like(arr, dtype=np.float32)
    return (arr / norm).astype(np.float32)


def cosine01(a: np.ndarray, b: np.ndarray) -> float:
    av = l2_normalize(a)
    bv = l2_normalize(b)
    if av.size == 0 or bv.size == 0 or av.shape != bv.shape:
        return 0.0
    return float(np.clip(np.dot(av, bv), 0.0, 1.0))


def normalize_label(label: str) -> str:
    return re.sub(r"\s+", " ", str(label).lower().replace("_", " ").replace("-", " ")).strip()


def feature_sha(feature: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(feature, dtype=np.float32).tobytes()).hexdigest()


def hsv_histogram(rgb: np.ndarray, mask: np.ndarray, bins: tuple[int, int, int] = (12, 12, 12)) -> np.ndarray:
    rgb = np.asarray(rgb)
    mask = np.asarray(mask, dtype=bool)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("rgb must have shape (H, W, 3)")
    if mask.shape != rgb.shape[:2]:
        raise ValueError("mask must match rgb height/width")
    if int(mask.sum()) == 0:
        return np.zeros(int(np.prod(bins)), dtype=np.float32)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    pixels = hsv[mask]
    hist, _ = np.histogramdd(
        pixels.astype(np.float32),
        bins=bins,
        range=((0, 180), (0, 256), (0, 256)),
    )
    return l2_normalize(hist.astype(np.float32).reshape(-1))


def voxel_downsample_points_colors(
    points: np.ndarray,
    colors: np.ndarray | None = None,
    voxel_size: float = 0.0,
    *,
    max_points: int | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    pts = np.asarray(points, dtype=np.float32)
    cols = None if colors is None else np.asarray(colors, dtype=np.float32)
    if pts.size == 0 or voxel_size <= 0:
        out_pts = pts
        out_cols = cols
    else:
        coords = np.floor(pts / float(voxel_size)).astype(np.int64)
        unique, inverse = np.unique(coords, axis=0, return_inverse=True)
        sums = np.zeros((unique.shape[0], 3), dtype=np.float64)
        np.add.at(sums, inverse, pts)
        counts = np.bincount(inverse, minlength=unique.shape[0]).astype(np.float64)
        out_pts = (sums / np.clip(counts[:, None], 1.0, None)).astype(np.float32)
        out_cols = None
        if cols is not None:
            color_sums = np.zeros((unique.shape[0], 3), dtype=np.float64)
            np.add.at(color_sums, inverse, cols)
            out_cols = (color_sums / np.clip(counts[:, None], 1.0, None)).astype(np.float32)
    if max_points is not None and out_pts.shape[0] > int(max_points):
        idx = np.linspace(0, out_pts.shape[0] - 1, int(max_points)).round().astype(np.int64)
        out_pts = out_pts[idx]
        if out_cols is not None:
            out_cols = out_cols[idx]
    return out_pts.astype(np.float32), None if out_cols is None else np.clip(out_cols, 0, 255).astype(np.float32)
