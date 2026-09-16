from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from plyfile import PlyData
from scipy.optimize import linear_sum_assignment


PAPER_NODE_COUNTS = {"O": 224, "C": 94, "U": 404}
PAPER_NODE_TOTAL = 722
PAPER_EDGE_TOTAL = 592
PAPER_HIERARCHY_EDGE_TOTAL = 118
PAPER_TABLETOP_EDGE_TOTAL = 140

# This is the category list used by the OpenFunGraph++ benchmark code.  Unlike
# that code, labels containing alternatives separated by '/' are handled as
# alternatives.  This recovers the paper's 140-edge tabletop subset exactly.
TABLETOP_OBJECT_LABELS = frozenset(
    {
        "bottle",
        "condiment",
        "pump bottle",
        "jar",
        "cup",
        "microwave",
        "rice cooker",
        "pot",
        "kettle",
        "coffee maker",
    }
)


@dataclass(frozen=True)
class EvalNode:
    node_id: str
    label: str
    role: str
    bbox_min: np.ndarray
    bbox_max: np.ndarray
    scene_id: str


@dataclass(frozen=True)
class EvalEdge:
    edge_id: str
    child_id: str
    parent_id: str
    relation: str
    scene_id: str


@dataclass
class BenchmarkGT:
    nodes_by_scene: dict[str, list[dict[str, Any]]]
    edges_by_scene: dict[str, list[dict[str, Any]]]
    roles: dict[str, str]
    hierarchy_edge_ids: set[str]
    tabletop_node_ids: set[str]
    tabletop_edge_ids: set[str]
    metadata: dict[str, Any]


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_label(label: str) -> str:
    return " ".join(str(label or "").lower().replace("_", " ").replace("-", " ").split())


def label_variants(label: str) -> set[str]:
    return {normalize_label(part) for part in str(label or "").split("/") if normalize_label(part)}


def read_fungraph3d_split(path: str | Path) -> list[str]:
    sequences = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for raw in handle:
            sequence = raw.strip()
            if sequence and not sequence.startswith("#"):
                sequences.append(sequence)
    if not sequences:
        raise ValueError(f"empty FunGraph3D split: {path}")
    return sequences


def _relation_key(edge: dict[str, Any]) -> str:
    relation_id = str(edge.get("relation_id", "")).strip()
    if relation_id:
        return relation_id
    return "|".join(
        (
            str(edge["scene_id"]),
            str(edge["first_node_annot_id"]),
            str(edge["second_node_annot_id"]),
            str(edge.get("description", "")),
        )
    )


def infer_uco_roles(
    annotations: Sequence[dict[str, Any]],
    relations: Sequence[dict[str, Any]],
) -> tuple[dict[str, str], set[str]]:
    """Infer the benchmark's O/C/U roles from its closed hierarchy triangles.

    Relations use the annotation convention child -> parent.  A hierarchy is
    represented by U->C, C->O and the legacy direct U->O edge.  Nodes outside
    such triangles follow the original two-level convention: a node with a
    child is O, otherwise U.
    """

    node_ids = {str(node["annot_id"]) for node in annotations}
    parents: dict[str, set[str]] = defaultdict(set)
    parent_nodes: set[str] = set()
    edge_by_pair: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for edge in relations:
        child = str(edge["first_node_annot_id"])
        parent = str(edge["second_node_annot_id"])
        if child not in node_ids or parent not in node_ids or child == parent:
            continue
        parents[child].add(parent)
        parent_nodes.add(parent)
        edge_by_pair[(child, parent)].append(edge)

    roles: dict[str, str] = {}
    hierarchy_edge_ids: set[str] = set()
    for unit in sorted(node_ids):
        for carrier in sorted(parents.get(unit, ())):
            for obj in sorted(parents.get(unit, set()) & parents.get(carrier, set())):
                previous = (roles.get(unit), roles.get(carrier), roles.get(obj))
                expected = ("U", "C", "O")
                if any(old is not None and old != new for old, new in zip(previous, expected)):
                    raise ValueError(
                        "ambiguous O/C/U topology for hierarchy triangle "
                        f"({unit}, {carrier}, {obj}): existing roles={previous}"
                    )
                roles[unit], roles[carrier], roles[obj] = expected
                for edge in edge_by_pair[(unit, carrier)]:
                    hierarchy_edge_ids.add(_relation_key(edge))

    for node_id in node_ids:
        roles.setdefault(node_id, "O" if node_id in parent_nodes else "U")
    return roles, hierarchy_edge_ids


def _paper_annotation_snapshot(
    annotations: list[dict[str, Any]],
    relations: list[dict[str, Any]],
    scene_ids: set[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select the published 722-node snapshot, detecting the known local drift.

    The local combined annotation file contains the exact paper snapshot as its
    first 722 FunGraph3D records, followed by 28 unreferenced 0kitchen records.
    We only apply this compatibility path after validating every invariant; a
    differently edited/reordered file fails instead of silently truncating.
    """

    selected = [node for node in annotations if str(node.get("scene_id")) in scene_ids]
    metadata: dict[str, Any] = {
        "fungraph3d_nodes_in_source": len(selected),
        "paper_snapshot_compatibility_filter": False,
        "excluded_trailing_annotations": 0,
    }
    if len(selected) == PAPER_NODE_TOTAL:
        return selected, metadata
    if len(selected) < PAPER_NODE_TOTAL:
        raise ValueError(
            f"expanded GT has {len(selected)} FunGraph3D nodes; paper requires {PAPER_NODE_TOTAL}"
        )

    candidate = selected[:PAPER_NODE_TOTAL]
    tail = selected[PAPER_NODE_TOTAL:]
    candidate_ids = {str(node["annot_id"]) for node in candidate}
    tail_ids = {str(node["annot_id"]) for node in tail}
    relation_endpoints = {
        str(endpoint)
        for edge in relations
        if str(edge.get("scene_id")) in scene_ids
        for endpoint in (edge["first_node_annot_id"], edge["second_node_annot_id"])
    }
    candidate_relations = [
        edge
        for edge in relations
        if str(edge.get("scene_id")) in scene_ids
        and str(edge["first_node_annot_id"]) in candidate_ids
        and str(edge["second_node_annot_id"]) in candidate_ids
    ]
    roles, hierarchy_ids = infer_uco_roles(candidate, candidate_relations)
    role_counts = Counter(roles.values())
    safe_known_drift = (
        not (tail_ids & relation_endpoints)
        and len(candidate_relations) == PAPER_EDGE_TOTAL
        and dict(role_counts) == PAPER_NODE_COUNTS
        and len(hierarchy_ids) == PAPER_HIERARCHY_EDGE_TOTAL
    )
    if not safe_known_drift:
        raise ValueError(
            "expanded GT differs from the paper snapshot and cannot be safely resolved: "
            f"nodes={len(selected)}, candidate_roles={dict(role_counts)}, "
            f"candidate_edges={len(candidate_relations)}, tail_relation_endpoints="
            f"{len(tail_ids & relation_endpoints)}"
        )
    metadata.update(
        {
            "paper_snapshot_compatibility_filter": True,
            "excluded_trailing_annotations": len(tail),
            "excluded_annotation_ids": [str(node["annot_id"]) for node in tail],
            "compatibility_reason": (
                "source appends relation-free records after the exact 722-node published prefix"
            ),
        }
    )
    return candidate, metadata


def load_paper_benchmark_gt(
    *,
    annotations_path: str | Path,
    relations_path: str | Path,
    sequences: Sequence[str],
) -> BenchmarkGT:
    annotations_path = Path(annotations_path)
    relations_path = Path(relations_path)
    all_annotations = list(_read_json(annotations_path))
    all_relations = list(_read_json(relations_path))
    scene_ids = {sequence.split("/", 1)[0] for sequence in sequences}
    annotations, snapshot_meta = _paper_annotation_snapshot(
        all_annotations, all_relations, scene_ids
    )
    node_ids = {str(node["annot_id"]) for node in annotations}
    relations = [
        edge
        for edge in all_relations
        if str(edge.get("scene_id")) in scene_ids
        and str(edge["first_node_annot_id"]) in node_ids
        and str(edge["second_node_annot_id"]) in node_ids
    ]
    roles, hierarchy_edge_ids = infer_uco_roles(annotations, relations)

    children: dict[str, set[str]] = defaultdict(set)
    for edge in relations:
        children[str(edge["second_node_annot_id"])].add(str(edge["first_node_annot_id"]))
    tabletop_roots = {
        str(node["annot_id"])
        for node in annotations
        if label_variants(str(node.get("label", ""))) & TABLETOP_OBJECT_LABELS
    }
    tabletop_node_ids = set(tabletop_roots)
    frontier = list(tabletop_roots)
    while frontier:
        parent = frontier.pop()
        for child in children.get(parent, ()):
            if child not in tabletop_node_ids:
                tabletop_node_ids.add(child)
                frontier.append(child)
    tabletop_edge_ids = {
        _relation_key(edge)
        for edge in relations
        if str(edge["first_node_annot_id"]) in tabletop_node_ids
        and str(edge["second_node_annot_id"]) in tabletop_node_ids
    }

    role_counts = Counter(roles.values())
    observed = {
        "nodes": len(annotations),
        "edges": len(relations),
        "roles": dict(role_counts),
        "hierarchy_edges": len(hierarchy_edge_ids),
        "tabletop_edges": len(tabletop_edge_ids),
    }
    expected = {
        "nodes": PAPER_NODE_TOTAL,
        "edges": PAPER_EDGE_TOTAL,
        "roles": PAPER_NODE_COUNTS,
        "hierarchy_edges": PAPER_HIERARCHY_EDGE_TOTAL,
        "tabletop_edges": PAPER_TABLETOP_EDGE_TOTAL,
    }
    if observed != expected:
        raise ValueError(f"GT does not reproduce the paper benchmark: observed={observed}, expected={expected}")

    nodes_by_scene: dict[str, list[dict[str, Any]]] = defaultdict(list)
    edges_by_scene: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for node in annotations:
        nodes_by_scene[str(node["scene_id"])].append(node)
    for edge in relations:
        edges_by_scene[str(edge["scene_id"])].append(edge)
    metadata = {
        **snapshot_meta,
        "annotations_path": str(annotations_path.resolve()),
        "relations_path": str(relations_path.resolve()),
        "annotations_sha256": _sha256(annotations_path),
        "relations_sha256": _sha256(relations_path),
        "sequences": list(sequences),
        "counts": observed,
        "tabletop_node_count": len(tabletop_node_ids),
        "tabletop_definition": (
            "slash-normalized OpenFunGraph++ tabletop object labels plus transitive functional descendants"
        ),
        "hierarchy_definition": "U-to-C edges in closed U-C-O triangles",
    }
    return BenchmarkGT(
        nodes_by_scene=dict(nodes_by_scene),
        edges_by_scene=dict(edges_by_scene),
        roles=roles,
        hierarchy_edge_ids=hierarchy_edge_ids,
        tabletop_node_ids=tabletop_node_ids,
        tabletop_edge_ids=tabletop_edge_ids,
        metadata=metadata,
    )


def _load_ply_xyz(path: Path) -> np.ndarray:
    vertex = PlyData.read(str(path))["vertex"]
    return np.column_stack((vertex["x"], vertex["y"], vertex["z"])).astype(np.float64)


def materialize_gt_scene(
    benchmark: BenchmarkGT,
    *,
    scene_root: str | Path,
    scene_id: str,
) -> tuple[list[EvalNode], list[EvalEdge]]:
    scene_root = Path(scene_root)
    ply_path = scene_root / scene_id / f"{scene_id}.ply"
    if not ply_path.is_file():
        raise FileNotFoundError(f"FunGraph3D scene point cloud not found: {ply_path}")
    scene_points = _load_ply_xyz(ply_path)
    nodes: list[EvalNode] = []
    for raw in benchmark.nodes_by_scene.get(scene_id, []):
        indices = np.asarray(raw["indices"], dtype=np.int64)
        if indices.size == 0 or int(indices.min()) < 0 or int(indices.max()) >= len(scene_points):
            raise ValueError(f"invalid GT point indices for annotation {raw['annot_id']}")
        points = scene_points[indices]
        node_id = str(raw["annot_id"])
        nodes.append(
            EvalNode(
                node_id=node_id,
                label=str(raw["label"]),
                role=benchmark.roles[node_id],
                bbox_min=points.min(axis=0),
                bbox_max=points.max(axis=0),
                scene_id=scene_id,
            )
        )
    edges = [
        EvalEdge(
            edge_id=_relation_key(raw),
            child_id=str(raw["first_node_annot_id"]),
            parent_id=str(raw["second_node_annot_id"]),
            relation=str(raw.get("description", "")),
            scene_id=scene_id,
        )
        for raw in benchmark.edges_by_scene.get(scene_id, [])
    ]
    return nodes, edges


def _transform_bounds(
    bbox_min: np.ndarray, bbox_max: np.ndarray, transform: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    corners = np.asarray(
        [
            [x, y, z, 1.0]
            for x in (bbox_min[0], bbox_max[0])
            for y in (bbox_min[1], bbox_max[1])
            for z in (bbox_min[2], bbox_max[2])
        ],
        dtype=np.float64,
    )
    mapped = (transform @ corners.T).T[:, :3]
    return mapped.min(axis=0), mapped.max(axis=0)


def load_hhofg_prediction(
    graph_path: str | Path,
    *,
    scene_id: str,
    transform: np.ndarray | None = None,
    incident_nodes_only: bool = True,
    atlas_path: str | Path | None = None,
) -> tuple[list[EvalNode], list[EvalEdge], dict[str, Any]]:
    graph_path = Path(graph_path)
    graph = _read_json(graph_path)
    map_dir = graph_path.parent
    map_json_path = map_dir / "map_nodes.json"
    map_npz_path = map_dir / "map_nodes.npz"
    if not map_json_path.is_file() or not map_npz_path.is_file():
        raise FileNotFoundError(
            f"prediction geometry requires {map_json_path.name} and {map_npz_path.name} beside the graph"
        )
    map_payload = _read_json(map_json_path)
    map_ids = [str(item) for item in map_payload["node_ids"]]
    map_metadata = {
        str(item["node_id"]): item for item in map_payload.get("nodes", [])
    }
    arrays = np.load(map_npz_path, allow_pickle=False)
    mins = np.asarray(arrays["bbox_min"], dtype=np.float64)
    maxs = np.asarray(arrays["bbox_max"], dtype=np.float64)
    if len(map_ids) != len(mins) or mins.shape != maxs.shape:
        raise ValueError("map_nodes JSON/NPZ geometry arrays are not aligned")
    geometry = {node_id: (mins[idx], maxs[idx]) for idx, node_id in enumerate(map_ids)}

    pending_edges = [e for e in graph.get("edges", []) if e.get("functional_status") == "pending"]
    raw_edges = [e for e in graph.get("edges", []) if e.get("functional_status") != "pending"]
    incident = {
        str(endpoint)
        for edge in raw_edges
        for endpoint in (edge["parent_node_id"], edge["child_node_id"])
    }
    nodes: list[EvalNode] = []
    for raw in graph.get("nodes", []):
        if isinstance(raw, str):
            node_id = raw
            raw = map_metadata.get(node_id, {})
        else:
            node_id = str(raw["node_id"])
        if incident_nodes_only and node_id not in incident:
            continue
        if node_id not in geometry:
            raise ValueError(f"final graph node {node_id} has no serialized map geometry")
        bbox_min, bbox_max = geometry[node_id]
        if transform is not None:
            bbox_min, bbox_max = _transform_bounds(bbox_min, bbox_max, transform)
        nodes.append(
            EvalNode(
                node_id=node_id,
                label=str(raw.get("label") or raw.get("top_label") or ""),
                role=str(raw.get("role", "")),
                bbox_min=bbox_min,
                bbox_max=bbox_max,
                scene_id=scene_id,
            )
        )
    edges = [
        EvalEdge(
            edge_id=f"{scene_id}:pred:{idx}",
            child_id=str(raw["child_node_id"]),
            parent_id=str(raw["parent_node_id"]),
            relation=str(
                raw.get("relation_text")
                or raw.get("description")
                or raw.get("relation")
                or ""
            ).strip(),
            scene_id=scene_id,
        )
        for idx, raw in enumerate(raw_edges)
    ]
    if atlas_path is None:
        config_path = map_dir.parent / "run_config.json"
        config = _read_json(config_path) if config_path.is_file() else {}
        frontend = config.get("frontend_run")
        if not frontend:
            raise ValueError(f"U-O evaluation needs an atlas path for {graph_path}")
        atlas_path = Path(frontend) / "global_atlas.json"
    atlas = _read_json(Path(atlas_path))
    edges = complete_evaluation_edges(nodes, edges, raw_edges, atlas)
    virtual_count = len(edges) - len(raw_edges)
    metadata = {
        "graph_path": str(graph_path.resolve()),
        "graph_sha256": _sha256(graph_path),
        "graph_complete": bool(graph.get("graph_complete", False)),
        "predicted_nodes": len(nodes),
        "all_final_nodes": len(graph.get("nodes", [])),
        "predicted_edges": len(edges),
        "pending_structural_edges_excluded": len(pending_edges),
        "edges_with_relation_text": sum(bool(edge.relation) for edge in edges),
        "incident_nodes_only": incident_nodes_only,
        "transform_applied": transform is not None,
        "edge_protocol": "functional_triplet",
        "virtual_uo_edges": virtual_count,
        "atlas_path": str(atlas_path) if atlas_path is not None else None,
    }
    return nodes, edges, metadata


def complete_evaluation_edges(
    nodes: Sequence[EvalNode],
    edges: Sequence[EvalEdge],
    raw_edges: Sequence[dict[str, Any]],
    atlas: dict[str, Any],
) -> list[EvalEdge]:
    """Evaluation-only U-C-O closure; never alter a stored prediction graph."""
    from hhofg.mapping3d.semantic_candidate_builder import labels_compatible

    result = [
        replace(edge, child_id=edge.parent_id, parent_id=edge.child_id)
        if raw.get("edge_type") == "remote" else edge
        for edge, raw in zip(edges, raw_edges, strict=True)
    ]
    by_id = {node.node_id: node for node in nodes}
    owners = {str(e["child_node_id"]): str(e["parent_node_id"])
              for e in raw_edges if e.get("edge_type") == "O-C"}
    existing = {(e.child_id, e.parent_id) for e in result}

    def compatible(a: str, b: str) -> bool:
        return labels_compatible(a, b) or labels_compatible(b, a)

    for index, raw in enumerate(raw_edges):
        if raw.get("edge_type") != "C-U":
            continue
        carrier, unit = str(raw["parent_node_id"]), str(raw["child_node_id"])
        owner = owners.get(carrier)
        if owner is None or (unit, owner) in existing:
            continue
        if any(key not in by_id for key in (owner, carrier, unit)):
            raise ValueError("U-C-O closure references a missing prediction node")
        predicates = {
            str(u.get("ou_relation", "")).strip()
            for o in atlas.get("objects", [])
            if compatible(o.get("object", ""), by_id[owner].label)
            for c in o.get("functional_carriers", [])
            if compatible(c.get("carrier", ""), by_id[carrier].label)
            for u in c.get("interactive_units", [])
            if compatible(u.get("unit", ""), by_id[unit].label)
            and str(u.get("ou_relation", "")).strip()
        }
        predicate = next(iter(predicates)) if len(predicates) == 1 else edges[index].relation
        result.append(EvalEdge(f"{by_id[unit].scene_id}:virtual:{index}", unit, owner,
                               predicate, by_id[unit].scene_id))
        existing.add((unit, owner))
    return result


def aabb_iou(first: EvalNode, second: EvalNode) -> float:
    overlap = np.maximum(np.minimum(first.bbox_max, second.bbox_max) - np.maximum(first.bbox_min, second.bbox_min), 0.0)
    intersection = float(np.prod(overlap))
    first_volume = float(np.prod(np.maximum(first.bbox_max - first.bbox_min, 0.0)))
    second_volume = float(np.prod(np.maximum(second.bbox_max - second.bbox_min, 0.0)))
    union = first_volume + second_volume - intersection
    return intersection / union if union > 0.0 else 0.0


class SemanticRetrieval:
    """Paper-compatible CLIP/BERT retrieval backed by frozen GT vocabularies."""

    def __init__(
        self,
        *,
        label_vocabulary: Sequence[str],
        label_embeddings: np.ndarray,
        relation_vocabulary: Sequence[str],
        relation_embeddings: np.ndarray,
        predicted_label_embeddings: dict[str, np.ndarray],
        predicted_relation_embeddings: dict[str, np.ndarray],
    ) -> None:
        self.labels = list(label_vocabulary)
        self.relations = list(relation_vocabulary)
        self.label_index = {label: idx for idx, label in enumerate(self.labels)}
        self.relation_index = {relation: idx for idx, relation in enumerate(self.relations)}
        self.label_embeddings = self._normalize_rows(label_embeddings)
        self.relation_embeddings = self._normalize_rows(relation_embeddings)
        self.predicted_label_embeddings = {
            text: self._normalize_vector(value) for text, value in predicted_label_embeddings.items()
        }
        self.predicted_relation_embeddings = {
            text: self._normalize_vector(value) for text, value in predicted_relation_embeddings.items()
        }
        if len(self.labels) != len(self.label_embeddings):
            raise ValueError("label vocabulary and embedding row counts differ")
        if len(self.relations) != len(self.relation_embeddings):
            raise ValueError("relation vocabulary and embedding row counts differ")

    @staticmethod
    def _normalize_rows(values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        norms = np.linalg.norm(values, axis=1, keepdims=True)
        return values / np.maximum(norms, 1e-12)

    @staticmethod
    def _normalize_vector(value: np.ndarray) -> np.ndarray:
        value = np.asarray(value, dtype=np.float64).reshape(-1)
        return value / max(float(np.linalg.norm(value)), 1e-12)

    @staticmethod
    def _hit(
        query: str,
        target: str,
        *,
        vocabulary: Sequence[str],
        target_index: dict[str, int],
        vocabulary_embeddings: np.ndarray,
        query_embeddings: dict[str, np.ndarray],
        topk: int,
        threshold: float,
    ) -> tuple[bool, float, int | None]:
        if not query or query not in query_embeddings or target not in target_index:
            return False, -1.0, None
        similarities = query_embeddings[query] @ vocabulary_embeddings.T
        target_similarity = float(similarities[target_index[target]])
        order = np.argsort(similarities)[::-1]
        rank_positions = np.flatnonzero(order == target_index[target])
        rank = int(rank_positions[0]) + 1 if rank_positions.size else None
        return bool((rank is not None and rank <= topk) or target_similarity > threshold), target_similarity, rank

    def label_hit(self, query: str, target: str, *, topk: int, threshold: float) -> tuple[bool, float, int | None]:
        return self._hit(
            query,
            target,
            vocabulary=self.labels,
            target_index=self.label_index,
            vocabulary_embeddings=self.label_embeddings,
            query_embeddings=self.predicted_label_embeddings,
            topk=topk,
            threshold=threshold,
        )

    def relation_hit(self, query: str, target: str, *, topk: int, threshold: float) -> tuple[bool, float, int | None]:
        return self._hit(
            query,
            target,
            vocabulary=self.relations,
            target_index=self.relation_index,
            vocabulary_embeddings=self.relation_embeddings,
            query_embeddings=self.predicted_relation_embeddings,
            topk=topk,
            threshold=threshold,
        )


def _hungarian_hits(quality: np.ndarray, valid: np.ndarray) -> list[tuple[int, int]]:
    if quality.size == 0 or not bool(valid.any()):
        return []
    cardinality_weight = float(max(quality.shape) + 1)
    cost = np.zeros_like(quality, dtype=np.float64)
    cost[valid] = -cardinality_weight - quality[valid]
    rows, cols = linear_sum_assignment(cost)
    return [(int(row), int(col)) for row, col in zip(rows, cols) if valid[row, col]]


def evaluate_nodes(
    gt_nodes: Sequence[EvalNode],
    pred_nodes: Sequence[EvalNode],
    retrieval: SemanticRetrieval,
    *,
    topk: int = 3,
    threshold: float = 0.75,
    require_same_role: bool = False,
) -> dict[str, Any]:
    quality = np.zeros((len(gt_nodes), len(pred_nodes)), dtype=np.float64)
    valid = np.zeros_like(quality, dtype=bool)
    geometry_pairs = 0
    semantic_pairs = 0
    has_geometry = np.zeros(len(gt_nodes), dtype=bool)
    has_semantics = np.zeros(len(gt_nodes), dtype=bool)
    for gt_idx, gt in enumerate(gt_nodes):
        for pred_idx, pred in enumerate(pred_nodes):
            iou = aabb_iou(gt, pred)
            if iou <= 0.0:
                continue
            geometry_pairs += 1
            has_geometry[gt_idx] = True
            semantic_ok, _, _ = retrieval.label_hit(
                pred.label, gt.label, topk=topk, threshold=threshold
            )
            if not semantic_ok:
                continue
            semantic_pairs += 1
            has_semantics[gt_idx] = True
            if require_same_role and gt.role != pred.role:
                continue
            valid[gt_idx, pred_idx] = True
            quality[gt_idx, pred_idx] = iou
    matches = _hungarian_hits(quality, valid)
    matched_gt = {gt_idx for gt_idx, _ in matches}
    failure_reasons = Counter()
    unmatched_gt = []
    for gt_idx, gt in enumerate(gt_nodes):
        if gt_idx in matched_gt:
            continue
        if not has_geometry[gt_idx]:
            reason = "no_positive_3d_iou"
        elif not has_semantics[gt_idx]:
            reason = "label_retrieval_miss"
        else:
            reason = "one_to_one_assignment_conflict"
        failure_reasons[reason] += 1
        unmatched_gt.append(
            {"gt_node_id": gt.node_id, "label": gt.label, "role": gt.role, "reason": reason}
        )
    return {
        "gt_count": len(gt_nodes),
        "pred_count": len(pred_nodes),
        "matched_count": len(matches),
        "recall": len(matches) / len(gt_nodes) if gt_nodes else 0.0,
        "geometry_candidate_pairs": geometry_pairs,
        "semantic_candidate_pairs": semantic_pairs,
        "failure_reasons": dict(sorted(failure_reasons.items())),
        "unmatched_gt": unmatched_gt,
        "matches": [
            {
                "gt_node_id": gt_nodes[gt_idx].node_id,
                "pred_node_id": pred_nodes[pred_idx].node_id,
                "iou": float(quality[gt_idx, pred_idx]),
            }
            for gt_idx, pred_idx in matches
        ],
    }


def evaluate_edges(
    gt_edges: Sequence[EvalEdge],
    pred_edges: Sequence[EvalEdge],
    gt_nodes_by_id: dict[str, EvalNode],
    pred_nodes_by_id: dict[str, EvalNode],
    retrieval: SemanticRetrieval,
    *,
    topk: int = 5,
    threshold: float = 0.70,
) -> dict[str, Any]:
    endpoint_quality = np.zeros((len(gt_edges), len(pred_edges)), dtype=np.float64)
    endpoint_valid = np.zeros_like(endpoint_quality, dtype=bool)
    triplet_valid = np.zeros_like(endpoint_quality, dtype=bool)
    missing_predicate_candidates = 0
    predicate_mismatch_candidates = 0
    has_geometry = np.zeros(len(gt_edges), dtype=bool)
    has_endpoint_semantics = np.zeros(len(gt_edges), dtype=bool)
    has_nonempty_predicate = np.zeros(len(gt_edges), dtype=bool)
    has_predicate_semantics = np.zeros(len(gt_edges), dtype=bool)
    for gt_idx, gt_edge in enumerate(gt_edges):
        gt_child = gt_nodes_by_id[gt_edge.child_id]
        gt_parent = gt_nodes_by_id[gt_edge.parent_id]
        for pred_idx, pred_edge in enumerate(pred_edges):
            pred_child = pred_nodes_by_id.get(pred_edge.child_id)
            pred_parent = pred_nodes_by_id.get(pred_edge.parent_id)
            if pred_child is None or pred_parent is None:
                continue
            child_iou = aabb_iou(gt_child, pred_child)
            parent_iou = aabb_iou(gt_parent, pred_parent)
            if child_iou <= 0.0 or parent_iou <= 0.0:
                continue
            has_geometry[gt_idx] = True
            child_ok, _, _ = retrieval.label_hit(
                pred_child.label, gt_child.label, topk=topk, threshold=threshold
            )
            parent_ok, _, _ = retrieval.label_hit(
                pred_parent.label, gt_parent.label, topk=topk, threshold=threshold
            )
            if not child_ok or not parent_ok:
                continue
            has_endpoint_semantics[gt_idx] = True
            endpoint_valid[gt_idx, pred_idx] = True
            endpoint_quality[gt_idx, pred_idx] = 0.5 * (child_iou + parent_iou)
            if not pred_edge.relation:
                missing_predicate_candidates += 1
                continue
            has_nonempty_predicate[gt_idx] = True
            relation_ok, _, _ = retrieval.relation_hit(
                pred_edge.relation, gt_edge.relation, topk=topk, threshold=threshold
            )
            if relation_ok:
                triplet_valid[gt_idx, pred_idx] = True
                has_predicate_semantics[gt_idx] = True
            else:
                predicate_mismatch_candidates += 1

    endpoint_matches = _hungarian_hits(endpoint_quality, endpoint_valid)
    triplet_matches = _hungarian_hits(endpoint_quality, triplet_valid)
    endpoint_matched_gt = {gt_idx for gt_idx, _ in endpoint_matches}
    triplet_matched_gt = {gt_idx for gt_idx, _ in triplet_matches}
    endpoint_failures = Counter()
    triplet_failures = Counter()
    unmatched_triplets = []
    for gt_idx, edge in enumerate(gt_edges):
        if gt_idx not in endpoint_matched_gt:
            endpoint_reason = (
                "no_endpoint_pair_with_positive_3d_iou"
                if not has_geometry[gt_idx]
                else "endpoint_label_retrieval_miss"
                if not has_endpoint_semantics[gt_idx]
                else "one_to_one_edge_assignment_conflict"
            )
            endpoint_failures[endpoint_reason] += 1
        if gt_idx in triplet_matched_gt:
            continue
        if not has_geometry[gt_idx]:
            reason = "no_endpoint_pair_with_positive_3d_iou"
        elif not has_endpoint_semantics[gt_idx]:
            reason = "endpoint_label_retrieval_miss"
        elif not has_nonempty_predicate[gt_idx]:
            reason = "prediction_relation_text_missing"
        elif not has_predicate_semantics[gt_idx]:
            reason = "predicate_retrieval_miss"
        else:
            reason = "one_to_one_edge_assignment_conflict"
        triplet_failures[reason] += 1
        unmatched_triplets.append(
            {
                "gt_edge_id": edge.edge_id,
                "child_id": edge.child_id,
                "parent_id": edge.parent_id,
                "relation": edge.relation,
                "reason": reason,
            }
        )
    total = len(gt_edges)
    return {
        "gt_count": total,
        "pred_count": len(pred_edges),
        "endpoint_matched_count": len(endpoint_matches),
        "endpoint_edge_recall": len(endpoint_matches) / total if total else 0.0,
        "triplet_matched_count": len(triplet_matches),
        "triplet_recall": len(triplet_matches) / total if total else 0.0,
        "predicate_accuracy_given_recalled_endpoints": (
            len(triplet_matches) / len(endpoint_matches) if endpoint_matches else 0.0
        ),
        "missing_predicate_candidate_pairs": missing_predicate_candidates,
        "predicate_mismatch_candidate_pairs": predicate_mismatch_candidates,
        "endpoint_failure_reasons": dict(sorted(endpoint_failures.items())),
        "triplet_failure_reasons": dict(sorted(triplet_failures.items())),
        "unmatched_triplets": unmatched_triplets,
        "triplet_matches": [
            {
                "gt_edge_id": gt_edges[gt_idx].edge_id,
                "pred_edge_id": pred_edges[pred_idx].edge_id,
                "endpoint_mean_iou": float(endpoint_quality[gt_idx, pred_idx]),
            }
            for gt_idx, pred_idx in triplet_matches
        ],
    }


def _sum_metric(records: Iterable[dict[str, Any]], matched_key: str, total_key: str) -> dict[str, Any]:
    records = list(records)
    matched = sum(int(record[matched_key]) for record in records)
    total = sum(int(record[total_key]) for record in records)
    return {"matched_count": matched, "gt_count": total, "recall": matched / total if total else 0.0}


def _sum_counters(records: Iterable[dict[str, Any]], key: str) -> dict[str, int]:
    total: Counter[str] = Counter()
    for record in records:
        total.update({str(name): int(count) for name, count in record.get(key, {}).items()})
    return dict(sorted(total.items()))


def evaluate_fungraph3d(
    *,
    benchmark: BenchmarkGT,
    scene_root: str | Path,
    sequence_predictions: dict[str, Path],
    retrieval: SemanticRetrieval,
    transform_by_sequence: dict[str, np.ndarray] | None = None,
    atlas_by_sequence: dict[str, Path] | None = None,
) -> dict[str, Any]:
    per_sequence: dict[str, Any] = {}
    transform_by_sequence = transform_by_sequence or {}
    for sequence, graph_path in sequence_predictions.items():
        scene_id = sequence.split("/", 1)[0]
        gt_nodes, gt_edges = materialize_gt_scene(
            benchmark, scene_root=scene_root, scene_id=scene_id
        )
        pred_nodes, pred_edges, prediction_meta = load_hhofg_prediction(
            graph_path,
            scene_id=scene_id,
            transform=transform_by_sequence.get(sequence),
            incident_nodes_only=True,
            atlas_path=(atlas_by_sequence or {}).get(sequence),
        )
        gt_by_id = {node.node_id: node for node in gt_nodes}
        pred_by_id = {node.node_id: node for node in pred_nodes}

        node_results: dict[str, Any] = {}
        for role in ("O", "C", "U"):
            subset = [node for node in gt_nodes if node.role == role]
            node_results[role] = evaluate_nodes(subset, pred_nodes, retrieval)
        node_results["tabletop"] = evaluate_nodes(
            [node for node in gt_nodes if node.node_id in benchmark.tabletop_node_ids],
            pred_nodes,
            retrieval,
        )
        node_results["overall"] = evaluate_nodes(gt_nodes, pred_nodes, retrieval)
        node_results["overall_role_strict_diagnostic"] = evaluate_nodes(
            gt_nodes, pred_nodes, retrieval, require_same_role=True
        )

        edge_results = {
            "hierarchy": evaluate_edges(
                [edge for edge in gt_edges if edge.edge_id in benchmark.hierarchy_edge_ids],
                pred_edges,
                gt_by_id,
                pred_by_id,
                retrieval,
            ),
            "tabletop": evaluate_edges(
                [edge for edge in gt_edges if edge.edge_id in benchmark.tabletop_edge_ids],
                pred_edges,
                gt_by_id,
                pred_by_id,
                retrieval,
            ),
            "overall": evaluate_edges(
                gt_edges, pred_edges, gt_by_id, pred_by_id, retrieval
            ),
        }
        per_sequence[sequence] = {
            "prediction": prediction_meta,
            "nodes": node_results,
            "edges": edge_results,
        }

    node_summary = {}
    for subset in ("O", "C", "U", "tabletop", "overall", "overall_role_strict_diagnostic"):
        records = [record["nodes"][subset] for record in per_sequence.values()]
        node_summary[subset] = {
            **_sum_metric(records, "matched_count", "gt_count"),
            "failure_reasons": _sum_counters(records, "failure_reasons"),
        }
    edge_summary = {}
    for subset in ("hierarchy", "tabletop", "overall"):
        records = [record["edges"][subset] for record in per_sequence.values()]
        edge_summary[subset] = {
            "triplet": {
                **_sum_metric(records, "triplet_matched_count", "gt_count"),
                "failure_reasons": _sum_counters(records, "triplet_failure_reasons"),
            },
            "endpoint_only_diagnostic": {
                **_sum_metric(records, "endpoint_matched_count", "gt_count"),
                "failure_reasons": _sum_counters(records, "endpoint_failure_reasons"),
            },
        }
    return {
        "protocol": {
            "geometry": "axis-aligned 3D box IoU > 0",
            "node_semantics": "CLIP ViT-B/16 Top-3 OR cosine > 0.75",
            "triplet_semantics": "CLIP endpoints and BERT predicate Top-5 OR cosine > 0.70",
            "matching": "subset-wise Hungarian; maximum cardinality then maximum IoU",
            "aggregation": "dataset micro recall (sum hits / sum GT)",
            "prediction_node_pool": "nodes incident to final functional graph edges",
            "prediction_role_gate": False,
            "edge_protocol": "functional_triplet",
            "virtual_uo_closure": True,
            "edge_direction": (
                "local child -> parent; remote source -> target"
            ),
            "paper_ambiguities": [
                "paper does not publish the keyframe downsampling schedule",
                "paper does not specify Hungarian cost; valid match cardinality is prioritized before IoU",
                "tabletop membership is reconstructed from the referenced implementation vocabulary",
            ],
        },
        "benchmark": benchmark.metadata,
        "evaluated_sequences": list(sequence_predictions),
        "missing_sequences": [
            sequence
            for sequence in benchmark.metadata["sequences"]
            if sequence not in sequence_predictions
        ],
        "per_sequence": per_sequence,
        "summary": {"nodes": node_summary, "edges": edge_summary},
    }
