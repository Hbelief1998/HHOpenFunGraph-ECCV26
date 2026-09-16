from __future__ import annotations

import cv2
import numpy as np


def transform_translation(T: np.ndarray) -> np.ndarray:
    return np.asarray(T, dtype=np.float64)[:3, 3]


def rigid_interp_split(t: float, T0: np.ndarray, t0: float, T1: np.ndarray, t1: float) -> np.ndarray:
    if t1 == t0:
        return np.asarray(T0, dtype=np.float64).copy()
    alpha = float((t - t0) / (t1 - t0))
    T0 = np.asarray(T0, dtype=np.float64)
    T1 = np.asarray(T1, dtype=np.float64)
    r0, _ = cv2.Rodrigues(T0[:3, :3])
    r1, _ = cv2.Rodrigues(T1[:3, :3])
    r = (1.0 - alpha) * r0.reshape(3) + alpha * r1.reshape(3)
    R, _ = cv2.Rodrigues(r)
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = R
    out[:3, 3] = (1.0 - alpha) * T0[:3, 3] + alpha * T1[:3, 3]
    return out


def nearest_or_interpolated_pose(
    timestamp: str | float,
    poses_by_timestamp: dict[str, np.ndarray],
    *,
    time_distance_threshold: float = np.inf,
    use_interpolation: bool = True,
) -> np.ndarray | None:
    if not poses_by_timestamp:
        return None
    desired = float(timestamp)
    keys = sorted(float(k) for k in poses_by_timestamp.keys())
    if desired < keys[0] or desired > keys[-1]:
        return None
    key_map = {float(k): k for k in poses_by_timestamp.keys()}
    if desired in key_map:
        return poses_by_timestamp[key_map[desired]]
    if not use_interpolation:
        nearest = min(keys, key=lambda k: abs(k - desired))
        if abs(nearest - desired) > time_distance_threshold:
            return None
        return poses_by_timestamp[key_map[nearest]]
    lower = max(k for k in keys if k < desired)
    upper = min(k for k in keys if k > desired)
    if abs(upper - desired) > time_distance_threshold or abs(lower - desired) > time_distance_threshold:
        return None
    return rigid_interp_split(desired, poses_by_timestamp[key_map[lower]], lower, poses_by_timestamp[key_map[upper]], upper)
