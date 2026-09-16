from __future__ import annotations

import numpy as np

from .features import voxel_downsample_points_colors


def depth_iqr_keep_mask(points_camera: np.ndarray, *, iqr_scale: float, min_keep_ratio: float) -> np.ndarray:
    points = np.asarray(points_camera, dtype=np.float32)
    n = int(points.shape[0])
    if n < 4:
        return np.ones(n, dtype=bool)
    z = points[:, 2]
    valid = np.isfinite(z)
    if int(valid.sum()) < 4:
        return np.ones(n, dtype=bool)
    zv = z[valid]
    q1, q3 = np.quantile(zv, [0.25, 0.75])
    iqr = float(q3 - q1)
    if not np.isfinite(iqr):
        return np.ones(n, dtype=bool)
    lower = float(q1) - float(iqr_scale) * iqr
    upper = float(q3) + float(iqr_scale) * iqr
    keep_valid = (zv >= lower) & (zv <= upper)
    keep = np.zeros(n, dtype=bool)
    keep[valid] = keep_valid
    keep_count = int(keep.sum())
    keep_ratio = keep_count / max(1, int(valid.sum()))
    if keep_count < 4 or keep_ratio < float(min_keep_ratio):
        return np.ones(n, dtype=bool)
    return keep


def dbscan_main_cluster_keep_mask(
    points: np.ndarray,
    *,
    eps: float,
    min_samples: int,
    min_keep_ratio: float,
    pre_voxel_size: float | None = None,
) -> tuple[np.ndarray, bool]:
    pts = np.asarray(points, dtype=np.float32)
    n = int(pts.shape[0])
    if n < max(2, int(min_samples)):
        return np.ones(n, dtype=bool), False
    sample = pts
    if pre_voxel_size is not None and pre_voxel_size > 0:
        sample, _ = voxel_downsample_points_colors(pts, None, float(pre_voxel_size))
        if sample.shape[0] < max(2, int(min_samples)):
            return np.ones(n, dtype=bool), False
    try:
        from sklearn.cluster import DBSCAN
    except Exception as exc:
        raise RuntimeError("scikit-learn is required for DBSCAN filtering") from exc
    labels = DBSCAN(eps=float(eps), min_samples=int(min_samples)).fit(sample).labels_
    valid = labels >= 0
    if int(valid.sum()) == 0:
        return np.ones(n, dtype=bool), True
    uniq, counts = np.unique(labels[valid], return_counts=True)
    main_label = uniq[np.argmax(counts)]
    sample_keep = labels == main_label
    if float(sample_keep.mean()) < float(min_keep_ratio):
        return np.ones(n, dtype=bool), True
    if sample.shape[0] == n:
        return sample_keep.astype(bool), True
    try:
        from scipy.spatial import cKDTree
    except Exception as exc:
        raise RuntimeError("SciPy is required for DBSCAN pre-voxel nearest-neighbor expansion") from exc
    tree = cKDTree(sample[sample_keep])
    dist, _ = tree.query(pts, k=1)
    keep = dist <= float(eps) * 1.5
    if float(keep.mean()) < float(min_keep_ratio) or int(keep.sum()) < max(2, int(min_samples)):
        return np.ones(n, dtype=bool), True
    return keep.astype(bool), True


def adaptive_dbscan_cluster_keep_mask(
    points: np.ndarray,
    pixels_uv: np.ndarray,
    *,
    center_uv: tuple[float, float],
    depth_median: float,
    fx: float,
    fy: float,
    parent_depth: float | None,
    parent_band_m: float | None,
    pixel_radius: float,
    eps_min: float,
    eps_max: float,
    min_samples: int,
    min_keep_points: int,
) -> tuple[np.ndarray, bool, dict]:
    pts = np.asarray(points, dtype=np.float32)
    uv = np.asarray(pixels_uv, dtype=np.float32)
    n = int(pts.shape[0])
    if n < max(2, int(min_samples)):
        return np.ones(n, dtype=bool), False, {"reason": "too_few_for_dbscan", "cluster_count": 1, "chosen_cluster_size": n}
    pixel_metric = float(depth_median) / max(1e-6, 0.5 * (float(fx) + float(fy)))
    eps = float(np.clip(float(pixel_radius) * pixel_metric, float(eps_min), float(eps_max)))
    try:
        from sklearn.cluster import DBSCAN
    except Exception as exc:
        raise RuntimeError("scikit-learn is required for DBSCAN filtering") from exc
    labels = DBSCAN(eps=eps, min_samples=int(min_samples)).fit(pts).labels_
    valid_labels = sorted(int(v) for v in np.unique(labels) if int(v) >= 0)
    if not valid_labels:
        return np.zeros(n, dtype=bool), True, {"reason": "no_dbscan_cluster", "eps": eps, "cluster_count": 0, "chosen_cluster_size": 0}
    center = np.asarray(center_uv, dtype=np.float32)
    candidates = []
    for lab in valid_labels:
        idx = labels == lab
        count = int(idx.sum())
        if count < int(min_keep_points):
            continue
        z = pts[idx, 2]
        med = float(np.median(z))
        mad = float(np.median(np.abs(z - med)))
        dist_uv = float(np.linalg.norm(np.median(uv[idx], axis=0) - center))
        parent_dist = None if parent_depth is None else abs(med - float(parent_depth))
        parent_ok = parent_dist is not None and parent_dist <= float(parent_band_m or 0.0)
        candidates.append(
            {
                "label": lab,
                "count": count,
                "median_depth": med,
                "mad": mad,
                "center_distance_px": dist_uv,
                "parent_distance_m": parent_dist,
                "parent_ok": bool(parent_ok),
            }
        )
    if not candidates:
        return np.zeros(n, dtype=bool), True, {"reason": "clusters_too_small", "eps": eps, "cluster_count": len(valid_labels), "chosen_cluster_size": 0}

    def rank(c: dict) -> tuple[int, float, float, int]:
        return (int(c["parent_ok"]), -float(c["center_distance_px"]), -float(c["mad"]), int(c["count"]))

    chosen = sorted(candidates, key=rank, reverse=True)[0]
    keep = labels == int(chosen["label"])
    return keep.astype(bool), True, {
        "reason": "chosen",
        "eps": eps,
        "cluster_count": len(valid_labels),
        "chosen_cluster_size": int(keep.sum()),
        "chosen_cluster_ratio": float(keep.mean()),
        "clusters": candidates,
        "chosen": chosen,
    }
