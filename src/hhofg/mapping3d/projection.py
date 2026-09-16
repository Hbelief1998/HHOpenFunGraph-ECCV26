from __future__ import annotations

import numpy as np


def box_iou_xyxy(a: list[float] | None, b: list[float] | None) -> float:
    if a is None or b is None:
        return 0.0
    ax1, ay1, ax2, ay2 = [float(v) for v in a]
    bx1, by1, bx2, by2 = [float(v) for v in b]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return 0.0 if union <= 1e-9 else float(inter / union)


def project_points_world_to_image(
    points_world: np.ndarray,
    K: np.ndarray,
    T_c2w: np.ndarray,
    image_height: int,
    image_width: int,
    *,
    near_margin_scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    pts = np.asarray(points_world, dtype=np.float32)
    K = np.asarray(K, dtype=np.float32)
    T_w2c = np.linalg.inv(np.asarray(T_c2w, dtype=np.float32))
    ones = np.ones((pts.shape[0], 1), dtype=np.float32)
    cam = (np.concatenate([pts, ones], axis=1) @ T_w2c.T)[:, :3]
    valid = np.isfinite(cam).all(axis=1) & (cam[:, 2] > 1e-6)
    if int(valid.sum()) == 0:
        return np.zeros((0, 2), dtype=np.float32), valid
    z = cam[:, 2]
    u = K[0, 0] * (cam[:, 0] / z) + K[0, 2]
    v = K[1, 1] * (cam[:, 1] / z) + K[1, 2]
    margin_x = float(image_width) * float(near_margin_scale)
    margin_y = float(image_height) * float(near_margin_scale)
    near = (u >= -margin_x) & (u <= float(image_width - 1) + margin_x) & (v >= -margin_y) & (v <= float(image_height - 1) + margin_y)
    valid = valid & near
    return np.stack([u[valid], v[valid]], axis=1).astype(np.float32), valid


def project_map_node_box(
    points_world: np.ndarray,
    K: np.ndarray,
    T_c2w: np.ndarray,
    image_height: int,
    image_width: int,
    *,
    min_projected_points: int = 3,
    depth_m: np.ndarray | None = None,
    depth_tolerance_m: float = 0.08,
    percentile_low: float = 0.0,
    percentile_high: float = 100.0,
) -> list[float] | None:
    uv, valid = project_points_world_to_image(points_world, K, T_c2w, image_height, image_width)
    if uv.shape[0] < int(min_projected_points):
        return None
    if depth_m is not None:
        pts = np.asarray(points_world, dtype=np.float32)
        T_w2c = np.linalg.inv(np.asarray(T_c2w, dtype=np.float32))
        cam = (np.concatenate([pts, np.ones((pts.shape[0], 1), dtype=np.float32)], axis=1) @ T_w2c.T)[:, :3]
        z = cam[valid, 2]
        u_int = np.clip(np.round(uv[:, 0]).astype(np.int64), 0, int(image_width) - 1)
        v_int = np.clip(np.round(uv[:, 1]).astype(np.int64), 0, int(image_height) - 1)
        depth_here = np.asarray(depth_m, dtype=np.float32)[v_int, u_int]
        visible = np.isfinite(depth_here) & (depth_here > 0) & (z <= depth_here + float(depth_tolerance_m))
        if int(visible.sum()) >= int(min_projected_points):
            uv = uv[visible]
    x = np.clip(uv[:, 0], 0.0, float(image_width - 1))
    y = np.clip(uv[:, 1], 0.0, float(image_height - 1))
    return [
        float(np.percentile(x, float(percentile_low))),
        float(np.percentile(y, float(percentile_low))),
        float(np.percentile(x, float(percentile_high))),
        float(np.percentile(y, float(percentile_high))),
    ]
