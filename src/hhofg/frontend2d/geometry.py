from __future__ import annotations

from typing import Optional


def box_touches_image_border(box_xyxy: Optional[list[float]], h: int, w: int, margin_px: int = 2) -> bool:
    if box_xyxy is None:
        return False
    x1, y1, x2, y2 = [float(v) for v in box_xyxy]
    margin = float(max(0, margin_px))
    return bool(x1 <= margin or y1 <= margin or x2 >= float(w - 1) - margin or y2 >= float(h - 1) - margin)


def box_iou_xyxy(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = [float(x) for x in a]
    bx1, by1, bx2, by2 = [float(x) for x in b]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return float(inter / denom) if denom > 0 else 0.0
