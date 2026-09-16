from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any, Iterable

from hhofg.core.types import Frame2DResult


VALID_SEMANTIC_SOURCES = {
    "atlas",
    "frame_result",
    "graph_frame_result",
    "allowed_parent_map",
}


def normalize_semantic_label(value: str | None) -> str:
    text = str(value or "").strip().lower().replace("_", " ").replace("-", " ")
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    return " ".join(text.split())


def labels_compatible(detection_label: str, semantic_label: str) -> bool:
    det = normalize_semantic_label(detection_label)
    sem = normalize_semantic_label(semantic_label)
    if not det or not sem:
        return False
    if det == sem:
        return True
    det_tokens = det.split()
    sem_tokens = sem.split()
    return (
        len(det_tokens) >= len(sem_tokens)
        and det_tokens[-len(sem_tokens) :] == sem_tokens
    )


def select_relation_text(texts: Iterable[str], *, labels: Iterable[str] = ()) -> str:
    """Accept conservative paraphrases; return an original witness, never a new predicate.

    Preserve negation, directions, arguments and action order. Only normalize
    articles, punctuation, a small action synonym set, and an explicit trailing
    endpoint name. Unknown paraphrases intentionally remain unresolved.
    """
    values = sorted({str(t).strip() for t in texts if t and str(t).strip()})
    targets = sorted({normalize_semantic_label(v) for v in labels if v}, key=len, reverse=True)

    def canonical(text: str) -> str:
        text = text.lower().replace("/", " or ")
        text = normalize_semantic_label(text)
        text = re.sub(r"\b(a|an|the)\b", " ", text)
        text = " ".join(text.split())
        for label in targets:
            if text.endswith(" " + label):
                text = text[:-(len(label) + 1)]
                break
        words = text.split()
        if words and words[0] in {"turn", "twist"} and (len(words) == 1 or words[1] not in {"on", "off"}):
            words[0] = "rotate"
        return " ".join(words)
    keys = {canonical(t) for t in values}
    return min(values, key=lambda t: (len(t), t)) if len(keys) == 1 and "" not in keys else ""


Pair = tuple[str, str]


@dataclass
class SemanticRelationIndex:
    """Typed semantic priors with their object-level ownership context.

    Pair membership answers whether a typed relation is legal.  Context keeps
    the reason why it is legal: an O-U may be a direct unit of the object or a
    fallback anchor for a unit that normally belongs to a carrier, while a C-U
    is scoped to the object entries that introduced that carrier relation.
    """

    oc_pairs: set[Pair] = field(default_factory=set)
    ou_pairs: set[Pair] = field(default_factory=set)
    cu_pairs: set[Pair] = field(default_factory=set)
    pair_sources: dict[str, dict[Pair, set[str]]] = field(
        default_factory=lambda: {"O-C": {}, "O-U": {}, "C-U": {}}
    )
    relation_evidence: dict[str, dict[Pair, set[tuple[str, str, str]]]] = field(
        default_factory=lambda: {"O-C": {}, "O-U": {}, "C-U": {}}
    )
    object_contexts: dict[str, dict[Pair, set[str]]] = field(
        default_factory=lambda: {"O-C": {}, "O-U": {}, "C-U": {}}
    )
    owner_kinds: dict[str, dict[Pair, set[str]]] = field(
        default_factory=lambda: {"O-C": {}, "O-U": {}, "C-U": {}}
    )

    def add(
        self,
        edge_type: str,
        parent_label: str | None,
        child_label: str | None,
        *,
        source: str,
        relation_text: str | None = None,
        object_context: str | None = None,
        owner_kind: str | None = None,
    ) -> None:
        if source not in VALID_SEMANTIC_SOURCES:
            raise ValueError(f"unsupported semantic source: {source}")
        pair = (
            normalize_semantic_label(parent_label),
            normalize_semantic_label(child_label),
        )
        if not all(pair):
            return
        pairs = {
            "O-C": self.oc_pairs,
            "O-U": self.ou_pairs,
            "C-U": self.cu_pairs,
        }.get(edge_type)
        if pairs is None:
            raise ValueError(f"unsupported semantic edge type: {edge_type}")
        pairs.add(pair)
        self.pair_sources[edge_type].setdefault(pair, set()).add(source)
        text = str(relation_text or "").strip()
        if text:
            self.relation_evidence[edge_type].setdefault(pair, set()).add(
                (text, normalize_semantic_label(object_context), source)
            )
        context = normalize_semantic_label(object_context)
        if context:
            self.object_contexts[edge_type].setdefault(pair, set()).add(context)
        kind = str(owner_kind or "").strip()
        if kind:
            self.owner_kinds[edge_type].setdefault(pair, set()).add(kind)

    def matching_pair(
        self, edge_type: str, parent_label: str, child_label: str
    ) -> Pair | None:
        matches = self.matching_pairs(edge_type, parent_label, child_label)
        return matches[0] if matches else None

    def matching_pairs(
        self, edge_type: str, parent_label: str, child_label: str
    ) -> list[Pair]:
        """Return every compatible prior pair, not an arbitrary first match.

        A label can occur in several object entries in the atlas.  Keeping all
        matches is important for hierarchy resolution because the same C-U
        pair (for example, door-handle) may be legal in several object
        contexts but not in the object instance currently being solved.
        """

        pairs = {
            "O-C": self.oc_pairs,
            "O-U": self.ou_pairs,
            "C-U": self.cu_pairs,
        }.get(edge_type, set())
        return [
            pair
            for pair in sorted(pairs)
            if labels_compatible(parent_label, pair[0])
            and labels_compatible(child_label, pair[1])
        ]

    def supports(self, edge_type: str, parent_label: str, child_label: str) -> bool:
        return self.matching_pair(edge_type, parent_label, child_label) is not None

    def sources_for(
        self, edge_type: str, parent_label: str, child_label: str
    ) -> set[str]:
        return set().union(*(self.pair_sources[edge_type].get(pair, set())
                             for pair in self.matching_pairs(edge_type, parent_label, child_label)))

    def relation_for(
        self, edge_type: str, parent_label: str, child_label: str,
        *, object_context: str | None = None,
    ) -> str:
        pairs = self.matching_pairs(edge_type, parent_label, child_label)
        if not pairs:
            return ""
        specificity = max(sum(len(label.split()) for label in pair) for pair in pairs)
        evidence = set().union(*(self.relation_evidence[edge_type].get(pair, set())
                                for pair in pairs
                                if sum(len(label.split()) for label in pair) == specificity))
        context = normalize_semantic_label(object_context)
        if context:
            scoped = {e for e in evidence if e[1] and
                      (labels_compatible(context, e[1]) or labels_compatible(e[1], context))}
            evidence = scoped or {e for e in evidence if not e[1]}
        observed = {e for e in evidence if e[2] in {"frame_result", "graph_frame_result"}}
        evidence = observed or evidence
        return select_relation_text((e[0] for e in evidence),
                                    labels=(parent_label, child_label, context))

    def metadata_for(
        self,
        edge_type: str,
        parent_label: str,
        child_label: str,
        *,
        object_context: str | None = None,
    ) -> dict[str, Any]:
        pairs = self.matching_pairs(edge_type, parent_label, child_label)
        context = normalize_semantic_label(object_context)
        if context:
            pairs = [
                pair
                for pair in pairs
                if any(
                    labels_compatible(context, value)
                    or labels_compatible(value, context)
                    for value in self.object_contexts[edge_type].get(pair, set())
                )
            ]
        if not pairs:
            return {}
        metadata = {
            "edge_type": edge_type,
            "semantic_pair": list(pairs[0]),
            "semantic_sources": sorted(
                set().union(
                    *(self.pair_sources[edge_type].get(pair, set()) for pair in pairs)
                )
            ),
            "object_contexts": sorted(
                set().union(
                    *(self.object_contexts[edge_type].get(pair, set()) for pair in pairs)
                )
            ),
            "owner_kinds": sorted(
                set().union(
                    *(self.owner_kinds[edge_type].get(pair, set()) for pair in pairs)
                )
            ),
        }
        evidence = set().union(*(self.relation_evidence[edge_type].get(pair, set()) for pair in pairs))
        if evidence:
            metadata["relation_evidence"] = [
                {"relation_text": text, "object_context": context, "source": source}
                for text, context, source in sorted(evidence)
            ]
        text = self.relation_for(edge_type, parent_label, child_label, object_context=object_context)
        if text:
            metadata["relation_text"] = text
        return metadata

    def parent_labels_for(
        self,
        edge_type: str,
        child_label: str,
        *,
        allowed_parent_labels: Iterable[str] | None = None,
    ) -> set[str]:
        """Return semantic parent labels compatible with a child label."""

        pairs = {
            "O-C": self.oc_pairs,
            "O-U": self.ou_pairs,
            "C-U": self.cu_pairs,
        }.get(edge_type, set())
        allowed = [normalize_semantic_label(v) for v in allowed_parent_labels or []]
        return {
            parent
            for parent, child in pairs
            if labels_compatible(child_label, child)
            and (
                not allowed
                or any(
                    labels_compatible(parent, value)
                    or labels_compatible(value, parent)
                    for value in allowed
                )
            )
        }


def _payload_objects(payload: dict[str, Any] | None) -> Iterable[dict[str, Any]]:
    if not isinstance(payload, dict):
        return []
    return [
        entry
        for entry in (payload.get("present") or []) + (payload.get("objects") or [])
        if isinstance(entry, dict)
    ]


def _add_payload(index: SemanticRelationIndex, payload: dict[str, Any] | None, source: str) -> None:
    if not isinstance(payload, dict):
        return
    for obj in _payload_objects(payload):
        object_label = obj.get("object")
        for carrier in obj.get("functional_carriers", []) or []:
            if not isinstance(carrier, dict):
                continue
            carrier_label = carrier.get("carrier")
            # O-C is an independent prior: it must survive even when the
            # carrier currently has no interactive unit.
            index.add(
                "O-C",
                object_label,
                carrier_label,
                source=source,
                relation_text=carrier.get("oc_relation"),
                object_context=object_label,
                owner_kind="carrier",
            )
            for unit in carrier.get("interactive_units", []) or []:
                if not isinstance(unit, dict):
                    continue
                unit_label = unit.get("unit")
                index.add(
                    "C-U",
                    carrier_label,
                    unit_label,
                    source=source,
                    relation_text=unit.get("cu_relation"),
                    object_context=object_label,
                    owner_kind="carrier",
                )
                index.add(
                    "O-U",
                    object_label,
                    unit_label,
                    source=source,
                    relation_text=unit.get("ou_relation"),
                    object_context=object_label,
                    owner_kind="carrier_fallback",
                )
        for unit in obj.get("direct_interactive_units", []) or []:
            if isinstance(unit, dict):
                index.add(
                    "O-U",
                    object_label,
                    unit.get("unit"),
                    source=source,
                    relation_text=unit.get("ou_relation"),
                    object_context=object_label,
                    owner_kind="direct",
                )

    for chain in payload.get("local_function_chains", []) or []:
        if not isinstance(chain, dict):
            continue
        index.add(
            "O-C",
            chain.get("object"),
            chain.get("carrier"),
            source=source,
            relation_text=chain.get("oc_relation"),
            object_context=chain.get("object"),
            owner_kind="carrier",
        )
        index.add(
            "C-U",
            chain.get("carrier"),
            chain.get("unit"),
            source=source,
            relation_text=chain.get("cu_relation"),
            object_context=chain.get("object"),
            owner_kind="carrier",
        )
        index.add(
            "O-U",
            chain.get("object"),
            chain.get("unit"),
            source=source,
            relation_text=chain.get("ou_relation"),
            object_context=chain.get("object"),
            owner_kind="carrier_fallback",
        )


def _add_allowed_parent_maps(index: SemanticRelationIndex, result: Frame2DResult) -> None:
    detections = list(result.detections)
    for child_label, parent_labels in (result.c_to_allowed_parents or {}).items():
        for parent_label in parent_labels or []:
            index.add(
                "O-C",
                parent_label,
                child_label,
                source="allowed_parent_map",
            )

    for child_label, parent_labels in (result.u_to_allowed_parents or {}).items():
        for parent_label in parent_labels or []:
            parent_roles = {
                det.role
                for det in detections
                if labels_compatible(det.label, str(parent_label))
            }
            parent_roles.update(
                str(role)
                for label, role in (result.label_to_role or {}).items()
                if labels_compatible(str(label), str(parent_label))
            )
            if "O" in parent_roles:
                index.add(
                    "O-U",
                    parent_label,
                    child_label,
                    source="allowed_parent_map",
                )
            if "C" in parent_roles:
                index.add(
                    "C-U",
                    parent_label,
                    child_label,
                    source="allowed_parent_map",
                )


def build_semantic_relation_index(
    result: Frame2DResult,
    global_atlas: dict[str, Any] | None = None,
) -> SemanticRelationIndex:
    index = SemanticRelationIndex()
    _add_payload(index, global_atlas, "atlas")
    _add_payload(index, result.frame_result, "frame_result")
    _add_payload(index, result.graph_frame_result, "graph_frame_result")
    _add_allowed_parent_maps(index, result)
    return index


def build_global_semantic_relation_index(
    global_atlas: dict[str, Any] | None,
) -> SemanticRelationIndex:
    """Build the sequence-level typed prior used by post-map hierarchy logic."""

    index = SemanticRelationIndex()
    _add_payload(index, global_atlas, "atlas")
    return index
