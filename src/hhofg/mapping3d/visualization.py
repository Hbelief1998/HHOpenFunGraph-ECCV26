from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from hhofg.core.serialization import write_json_atomic
from hhofg.core.types import FrameRecord

from .types import FrameAssociationResult, MapNode3D, Observation3D

ROLE_COLORS = {
    "O": (60, 120, 255),
    "C": (255, 160, 50),
    "U": (80, 220, 120),
}

ROLE_COLORS_RGB = {
    # Match OpenFunGraph++ TrackerColors: node_o, node_m, node_l.
    "O": np.array([255, 0, 0], dtype=np.uint8),
    "C": np.array([255, 255, 0], dtype=np.uint8),
    "U": np.array([0, 0, 255], dtype=np.uint8),
}

EDGE_COLORS_RGB = {
    # Match OpenFunGraph++ TrackerColors: link_om, link_ml, link_ol.
    "O-C": np.array([0, 255, 0], dtype=np.uint8),
    "C-U": np.array([128, 0, 128], dtype=np.uint8),
    "O-U": np.array([255, 165, 0], dtype=np.uint8),
    # Remote functional relation, distinct from hierarchy ownership edges.
    "remote": np.array([0, 200, 255], dtype=np.uint8),
}


def write_ply_points(path: str | Path, points: np.ndarray, colors: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pts = np.asarray(points, dtype=np.float32)
    cols = np.clip(np.asarray(colors, dtype=np.float32), 0, 255).astype(np.uint8)
    with path.open("w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {pts.shape[0]}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
        for p, c in zip(pts, cols):
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {int(c[0])} {int(c[1])} {int(c[2])}\n")


def write_map_plys(nodes: dict[str, MapNode3D], out_dir: str | Path) -> None:
    out_dir = Path(out_dir)
    ordered = [nodes[k] for k in sorted(nodes)]
    points = np.concatenate([n.points_world for n in ordered], axis=0) if ordered else np.zeros((0, 3), dtype=np.float32)
    colors = np.concatenate([n.colors_rgb for n in ordered], axis=0) if ordered else np.zeros((0, 3), dtype=np.float32)
    write_ply_points(out_dir / "map_points_rgb.ply", points, colors)
    centroids = np.stack([n.centroid_world for n in ordered], axis=0).astype(np.float32) if ordered else np.zeros((0, 3), dtype=np.float32)
    role_cols = np.asarray([ROLE_COLORS[n.role][::-1] for n in ordered], dtype=np.uint8) if ordered else np.zeros((0, 3), dtype=np.uint8)
    write_ply_points(out_dir / "map_node_centroids.ply", centroids, role_cols)
    from hhofg.core.serialization import write_json_atomic

    write_json_atomic(out_dir / "map_node_centroid_index.json", [{"point_index": i, "node_id": n.node_id} for i, n in enumerate(ordered)])


def infer_scene_base_ply_path(dataset_root: str | Path, sequence: str) -> Path | None:
    root = Path(dataset_root)
    parts = Path(sequence).parts
    if not parts:
        return None
    scene = parts[0]
    candidates = [
        root / scene / f"{scene}.ply",
        root / f"{scene}.ply",
        root / sequence / f"{scene}.ply",
    ]
    for p in candidates:
        if p.is_file():
            return p
    return None


def read_ply_points_rgb(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    try:
        from plyfile import PlyData
    except Exception as exc:
        raise RuntimeError("plyfile is required to read scene PLY files") from exc
    ply = PlyData.read(str(path))
    vertex = ply["vertex"].data
    points = np.stack([vertex[a] for a in ("x", "y", "z")], axis=1).astype(np.float32)
    if all(c in vertex.dtype.names for c in ("red", "green", "blue")):
        colors = np.stack([vertex[c] for c in ("red", "green", "blue")], axis=1).astype(np.uint8)
    else:
        colors = np.full((points.shape[0], 3), 180, dtype=np.uint8)
    finite = np.isfinite(points).all(axis=1)
    return points[finite], colors[finite]


def _subsample_scene(points: np.ndarray, colors: np.ndarray, keep_ratio: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    ratio = float(np.clip(keep_ratio, 0.0, 1.0))
    if points.shape[0] == 0 or ratio >= 0.999:
        return points, colors
    keep_n = max(1, int(round(points.shape[0] * ratio)))
    rng = np.random.default_rng(int(seed))
    idx = np.sort(rng.choice(points.shape[0], size=keep_n, replace=False))
    return points[idx], colors[idx]


def _fibonacci_sphere(center: np.ndarray, radius: float, samples: int) -> np.ndarray:
    if samples <= 1:
        return center.reshape(1, 3).astype(np.float32)
    i = np.arange(samples, dtype=np.float32)
    phi = np.pi * (3.0 - np.sqrt(5.0))
    y = 1.0 - (2.0 * i) / max(1.0, samples - 1.0)
    r = np.sqrt(np.clip(1.0 - y * y, 0.0, 1.0))
    theta = phi * i
    unit = np.stack([np.cos(theta) * r, y, np.sin(theta) * r], axis=1)
    return (center[None, :] + float(radius) * unit).astype(np.float32)


def _sample_ball_points(center: np.ndarray, radius: float, shell_samples: int = 192, shell_count: int = 4) -> np.ndarray:
    layers = [center.reshape(1, 3).astype(np.float32)]
    for shell_idx in range(1, int(shell_count) + 1):
        frac = shell_idx / float(shell_count)
        layers.append(_fibonacci_sphere(center, radius * frac, max(24, int(round(shell_samples * frac * frac)))))
    return np.concatenate(layers, axis=0).astype(np.float32)


def _sample_segment_points(start: np.ndarray, end: np.ndarray, spacing: float) -> np.ndarray:
    dist = float(np.linalg.norm(end - start))
    if dist <= 1e-9:
        return start.reshape(1, 3).astype(np.float32)
    n = max(2, int(np.ceil(dist / max(float(spacing), 1e-4))) + 1)
    t = np.linspace(0.0, 1.0, n, dtype=np.float32)
    return (start[None, :] * (1.0 - t[:, None]) + end[None, :] * t[:, None]).astype(np.float32)


def _orthonormal_basis(direction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    d = direction.astype(np.float32)
    d = d / max(float(np.linalg.norm(d)), 1e-9)
    ref = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    if abs(float(np.dot(d, ref))) > 0.9:
        ref = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    b1 = np.cross(d, ref)
    b1 = b1 / max(float(np.linalg.norm(b1)), 1e-9)
    b2 = np.cross(d, b1)
    b2 = b2 / max(float(np.linalg.norm(b2)), 1e-9)
    return b1.astype(np.float32), b2.astype(np.float32)


def _sample_tube_points(start: np.ndarray, end: np.ndarray, *, spacing: float, radius: float, ring_samples: int) -> np.ndarray:
    centers = _sample_segment_points(start, end, spacing)
    if centers.shape[0] == 1 or radius <= 1e-9 or ring_samples <= 2:
        return centers
    b1, b2 = _orthonormal_basis(end - start)
    angles = np.linspace(0.0, 2.0 * np.pi, int(ring_samples), endpoint=False, dtype=np.float32)
    parts = [centers]
    for radial in (0.55 * radius, radius):
        rings = [centers + radial * (np.cos(a) * b1 + np.sin(a) * b2)[None, :] for a in angles]
        parts.append(np.concatenate(rings, axis=0))
    return np.concatenate(parts, axis=0).astype(np.float32)


def _point_to_segment_dist2(points: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ab = b - a
    ab2 = float(np.dot(ab, ab))
    if ab2 <= 1e-12:
        return np.sum((points - a[None, :]) ** 2, axis=1)
    t = np.sum((points - a[None, :]) * ab[None, :], axis=1) / ab2
    t = np.clip(t, 0.0, 1.0)
    proj = a[None, :] + t[:, None] * ab[None, :]
    return np.sum((points - proj) ** 2, axis=1)


def _clear_scene_near_graph(
    points: np.ndarray,
    node_positions: dict[str, np.ndarray],
    edge_segments: list[tuple[np.ndarray, np.ndarray]],
    *,
    node_radius: float,
    edge_radius: float,
) -> np.ndarray:
    keep = np.ones(points.shape[0], dtype=bool)
    for center in node_positions.values():
        keep &= np.sum((points - center[None, :]) ** 2, axis=1) > node_radius * node_radius
    for a, b in edge_segments:
        keep &= _point_to_segment_dist2(points, a, b) > edge_radius * edge_radius
    return keep


def write_scene_graph_overlay_ply(
    *,
    scene_points: np.ndarray,
    scene_colors: np.ndarray,
    nodes: dict[str, MapNode3D],
    edges: list[dict],
    output_ply_path: str | Path,
    metadata_path: str | Path | None = None,
    scene_keep_ratio: float = 1.0,
    random_seed: int = 0,
    clear_scene_near_graph: bool = False,
    metadata_extra: dict | None = None,
) -> Path:
    scene_points = np.asarray(scene_points, dtype=np.float32)
    scene_colors = np.asarray(scene_colors, dtype=np.uint8)
    scene_points, scene_colors = _subsample_scene(scene_points, scene_colors, scene_keep_ratio, random_seed)
    if scene_points.shape[0] > 0:
        diag = float(np.linalg.norm(scene_points.max(axis=0) - scene_points.min(axis=0)))
    else:
        centers = np.stack([n.centroid_world for n in nodes.values()], axis=0) if nodes else np.zeros((1, 3), dtype=np.float32)
        diag = float(np.linalg.norm(centers.max(axis=0) - centers.min(axis=0)))
    node_radius = float(np.clip(diag * 0.0045, 0.01, 0.03))
    edge_radius = float(np.clip(node_radius * 0.45, 0.004, 0.015))
    edge_spacing = max(edge_radius * 0.6, 0.003)

    overlay_points: list[np.ndarray] = []
    overlay_colors: list[np.ndarray] = []
    node_positions = {node_id: node.centroid_world.astype(np.float32) for node_id, node in nodes.items()}
    # Derived objects can reuse a single member's geometry. Separate only
    # their display glyphs; serialized centroids and evaluation stay physical.
    display_offsets = {}
    for edge in edges:
        if edge.get("source") != "derived_object_completion":
            continue
        parent, child = str(edge["parent_node_id"]), str(edge["child_node_id"])
        if parent not in node_positions or child not in node_positions or parent in display_offsets:
            continue
        if np.linalg.norm(node_positions[parent] - node_positions[child]) < 2 * node_radius:
            offset = np.array([0.0, 0.0, 3.0 * node_radius], dtype=np.float32)
            node_positions[parent] = node_positions[parent] + offset
            display_offsets[parent] = offset.tolist()
    rendered_edges = []
    edge_segments: list[tuple[np.ndarray, np.ndarray]] = []

    for node_id in sorted(nodes):
        node = nodes[node_id]
        center = node_positions[node_id]
        sphere = _sample_ball_points(center, node_radius)
        color = ROLE_COLORS_RGB.get(node.role, np.array([255, 255, 255], dtype=np.uint8))
        overlay_points.append(sphere)
        overlay_colors.append(np.repeat(color[None, :], sphere.shape[0], axis=0))

    for edge in edges:
        src_id = str(edge["parent_node_id"])
        dst_id = str(edge["child_node_id"])
        src = node_positions.get(src_id)
        dst = node_positions.get(dst_id)
        if src is None or dst is None:
            continue
        color = EDGE_COLORS_RGB.get(str(edge["edge_type"]), np.array([255, 255, 255], dtype=np.uint8))
        tube = _sample_tube_points(src, dst, spacing=edge_spacing, radius=edge_radius, ring_samples=12)
        overlay_points.append(tube)
        overlay_colors.append(np.repeat(color[None, :], tube.shape[0], axis=0))
        edge_segments.append((src, dst))
        rendered_edges.append(edge)

    if clear_scene_near_graph and scene_points.shape[0] > 0:
        keep = _clear_scene_near_graph(
            scene_points,
            node_positions,
            edge_segments,
            node_radius=node_radius * 1.8,
            edge_radius=edge_radius * 1.6,
        )
        scene_points = scene_points[keep]
        scene_colors = scene_colors[keep]

    points = np.concatenate([scene_points, *overlay_points], axis=0) if overlay_points else scene_points
    colors = np.concatenate([scene_colors, *overlay_colors], axis=0) if overlay_colors else scene_colors
    output_ply_path = Path(output_ply_path)
    write_ply_points(output_ply_path, points, colors)
    if metadata_path is not None:
        payload = {
                "output_ply": str(output_ply_path),
                "scene_points_rendered": int(scene_points.shape[0]),
                "num_nodes_rendered": int(len(nodes)),
                "num_edges_rendered": int(len(rendered_edges)),
                "node_radius": node_radius,
                "display_offsets": display_offsets,
                "edge_radius": edge_radius,
                "nodes": [
                    {
                        "node_id": node_id,
                        "role": nodes[node_id].role,
                        "label": nodes[node_id].top_label,
                        "centroid_world": nodes[node_id].centroid_world.tolist(),
                        "display_centroid_world": node_positions[node_id].tolist(),
                    }
                    for node_id in sorted(nodes)
                ],
                "edges": rendered_edges,
        }
        if metadata_extra:
            payload.update(metadata_extra)
        write_json_atomic(metadata_path, payload)
    return output_ply_path


def draw_association_overlay(
    frame: FrameRecord,
    observations: list[Observation3D],
    assoc: FrameAssociationResult,
    out_path: str | Path,
) -> None:
    image = cv2.cvtColor(frame.rgb.copy(), cv2.COLOR_RGB2BGR)
    match_by_obs = {m["observation_id"]: m for m in assoc.matches}
    birth_by_obs = {b["observation_id"]: b for b in assoc.births}
    selected_pairs = {(p.observation_id, p.node_id): p for p in assoc.pair_scores if p.selected}
    for obs in observations:
        color = ROLE_COLORS.get(obs.role, (240, 240, 240))
        x1, y1, x2, y2 = [int(round(v)) for v in obs.box_xyxy]
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        label = "NEW"
        pair = None
        if obs.observation_id in match_by_obs:
            node_id = match_by_obs[obs.observation_id]["node_id"]
            label = node_id
            pair = selected_pairs.get((obs.observation_id, node_id))
        elif obs.observation_id in birth_by_obs:
            label = f"{birth_by_obs[obs.observation_id]['node_id']} NEW"
        if pair is not None and pair.projected_box_xyxy is not None:
            px1, py1, px2, py2 = [int(round(v)) for v in pair.projected_box_xyxy]
            cv2.rectangle(image, (px1, py1), (px2, py2), (255, 255, 255), 1)
            oc = (int((x1 + x2) / 2), int((y1 + y2) / 2))
            pc = (int((px1 + px2) / 2), int((py1 + py2) / 2))
            cv2.line(image, oc, pc, (255, 255, 255), 1)
            label = f"{label} s={pair.score_total:.2f} i/g/a/s={pair.score_iou:.2f}/{pair.score_geo:.2f}/{pair.score_app:.2f}/{pair.score_sem:.2f}"
        cv2.putText(image, label, (max(0, x1), max(16, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), image)
