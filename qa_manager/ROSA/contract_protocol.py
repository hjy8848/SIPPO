"""Typed, contract-preserving protocol helpers for DAGENT inference."""

from __future__ import annotations

import re
from typing import Any, Iterable, Mapping, Sequence

from qa_manager.ROSA.protocol import find_reference_ids, is_invalid_conclusion


CONTRACT_PROTOCOL_VERSION = "cp-dagent-v2-no-declared-out"
PLAN_RE = re.compile(r"<plan\b(?P<attrs>[^>]*)>(?P<body>.*?)</plan>", re.I | re.S)
NODE_RE = re.compile(
    r"<(?P<type>search|reason)\b(?P<attrs>[^>]*)>(?P<task>.*?)</(?P=type)>",
    re.I | re.S,
)
ATTR_RE = re.compile(r"([A-Za-z_][\w-]*)\s*=\s*([\"'])(.*?)\2", re.S)
REF_RE = re.compile(r"#(\d+)#")

SELECTION_OPS = {
    "argmax",
    "argmin",
    "closest",
    "earlier",
    "later",
    "longer",
    "newer",
    "older",
    "shorter",
    "younger",
}
BOOLEAN_OPS = {"equal", "not_equal"}
ALLOWED_OPS = SELECTION_OPS | BOOLEAN_OPS | {
    "apply_voucher",
    "arithmetic",
    "derive",
    "filter",
    "filter_constraints",
    "intersect",
    "intersect_product",
    "join_shop",
    "merge",
}


def _attrs(text: str) -> dict[str, str]:
    return {match.group(1).lower(): match.group(3).strip() for match in ATTR_RE.finditer(text)}


def _refs(text: str) -> list[str]:
    return REF_RE.findall(str(text or ""))


def _normalize_token(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(text or "").strip().lower()).strip("_")


def build_contract_planner_system_message(max_nodes: int = 5) -> str:
    limit = f"- Use at most {max_nodes} nodes.\n" if max_nodes > 0 else ""
    return f"""Generate a typed Search/Reason DAG for the question.

Output exactly one XML plan with this grammar:
<plan sink="#ID#" answer_type="TYPE">
<search id="ID" subject="ENTITY_OR_#ID#" relation="RELATION" candidate="OPTIONAL_ANSWER_CANDIDATE">ONE_ATOMIC_RETRIEVAL_QUESTION</search>
<reason id="ID" op="OPERATOR" inputs="#ID#,#ID#" return="candidate|value|boolean">CONCRETE_LOGIC_TASK</reason>
</plan>

Contract:
- IDs start at 1, increase by 1, and may reference only earlier nodes.
{limit}- Keep the original DAGENT roles: use Search for external facts and Reason for comparison, arithmetic, filtering, intersection, or merging.
- Every node must contribute to the single sink. Set sink to the only terminal node.
- For a dependent Search, put exactly one prior reference in subject and preserve the requested relation literally in relation.
- A Reason must list every input in inputs. Use one of: earlier, later, older, younger, longer, shorter, closest, argmin, argmax, equal, not_equal, arithmetic, filter, intersect, merge, derive.
- For a comparison that asks which named person, film, product, or other candidate wins, retrieve the compared attribute directly. Put the answer candidate represented by each Search in candidate, and use return="candidate".
- For yes/no comparison use return="boolean". Otherwise use return="value" only when the source value itself is the requested answer.
- Do not compare entity names when the question asks to compare dates, counts, prices, durations, or another attribute.
- Do not omit an external fact from the graph and expect Reason to retrieve it from memory.

Example chain:
<plan sink="#2#" answer_type="entity">
<search id="1" subject="Ada Lovelace" relation="husband">Who was Ada Lovelace's husband?</search>
<search id="2" subject="#1#" relation="father">Who was the father of #1#?</search>
</plan>

Example comparison:
<plan sink="#3#" answer_type="entity">
<search id="1" subject="Film A" relation="director_birth_year" candidate="Film A">When was the director of Film A born?</search>
<search id="2" subject="Film B" relation="director_birth_year" candidate="Film B">When was the director of Film B born?</search>
<reason id="3" op="later" inputs="#1#,#2#" return="candidate">Select the film whose director was born later using #1# and #2#.</reason>
</plan>
"""


def parse_contract_plan(text: str) -> dict[str, Any]:
    match = PLAN_RE.search(str(text or ""))
    if not match:
        return {"attrs": {}, "nodes": [], "parse_error": "missing <plan> block"}
    plan_attrs = _attrs(match.group("attrs"))
    nodes: list[dict[str, Any]] = []
    for node_match in NODE_RE.finditer(match.group("body")):
        attrs = _attrs(node_match.group("attrs"))
        # Legacy model outputs may still contain `out`; it is not part of the contract.
        attrs.pop("out", None)
        node_type = node_match.group("type").lower()
        node = {
            "type": node_type,
            "id": attrs.pop("id", ""),
            "task": " ".join(node_match.group("task").split()),
            **attrs,
        }
        if node_type == "reason":
            node["input_ids"] = _refs(node.get("inputs", ""))
        else:
            node["input_ids"] = _refs(node.get("subject", ""))
        nodes.append(node)
    return {"attrs": plan_attrs, "nodes": nodes, "parse_error": ""}


def _comparison_dimension(question: str) -> tuple[str, tuple[str, ...]] | None:
    question = str(question or "").lower()
    dimensions = (
        ("birth", ("born", "birth")),
        ("death", ("died", "death")),
        ("lifespan", ("lived longer", "lifespan", "age at death")),
        ("release", ("released", "release", "premiered", "published")),
        ("foundation", ("founded", "established", "formed")),
        ("population", ("population", "inhabitants")),
        ("distance", ("closer", "farther", "further", "distance")),
        ("duration", ("longer", "shorter", "duration", "running time", "runtime")),
        ("count", ("more", "fewer", "number of", "how many")),
    )
    for name, aliases in dimensions:
        if any(alias in question for alias in aliases):
            return name, aliases
    return None


def _relation_matches_dimension(relation: str, dimension: str) -> bool:
    relation = _normalize_token(relation)
    aliases = {
        "birth": ("birth", "born"),
        "death": ("death", "died"),
        "lifespan": ("lifespan", "lifetime", "age", "birth_death"),
        "release": ("release", "premiere", "publication"),
        "foundation": ("founded", "foundation", "established", "formed"),
        "population": ("population", "inhabitant"),
        "distance": ("distance", "location"),
        "duration": ("duration", "runtime", "running_time", "length"),
        "count": ("count", "number", "directors", "members", "episodes"),
    }
    return any(alias in relation for alias in aliases.get(dimension, (dimension,)))


def validate_contract_plan(
    question: str,
    plan: Mapping[str, Any],
    *,
    max_nodes: int = 0,
) -> list[str]:
    errors: list[str] = []
    nodes = list(plan.get("nodes") or [])
    attrs = dict(plan.get("attrs") or {})
    if plan.get("parse_error"):
        errors.append(str(plan["parse_error"]))
    if not nodes:
        errors.append("plan has no executable nodes")
        return errors
    if max_nodes > 0 and len(nodes) > max_nodes:
        errors.append(f"plan has {len(nodes)} nodes; maximum is {max_nodes}")

    ids = [str(node.get("id", "")) for node in nodes]
    if ids != [str(index) for index in range(1, len(nodes) + 1)]:
        errors.append("node IDs must be consecutive from 1")
    if len(set(ids)) != len(ids):
        errors.append("node IDs must be unique")

    by_id = {str(node.get("id", "")): node for node in nodes}
    children: dict[str, set[str]] = {node_id: set() for node_id in ids}
    for node in nodes:
        node_id = str(node.get("id", ""))
        node_type = str(node.get("type", ""))
        task = str(node.get("task", ""))
        if not task:
            errors.append(f"node {node_id or '?'} has no concrete task")
        input_ids = list(node.get("input_ids") or [])
        if node_type == "search":
            if not node.get("subject") or not node.get("relation"):
                errors.append(f"search node {node_id or '?'} needs subject and relation")
            if len(input_ids) > 1:
                errors.append(f"search node {node_id or '?'} may consume only one subject ref")
        elif node_type == "reason":
            op = _normalize_token(node.get("op", ""))
            if op not in ALLOWED_OPS:
                errors.append(f"reason node {node_id or '?'} has unsupported op={op or 'missing'}")
            if not input_ids:
                errors.append(f"reason node {node_id or '?'} has no typed inputs")
            return_kind = _normalize_token(node.get("return", ""))
            if op in SELECTION_OPS and return_kind != "candidate":
                errors.append(f"selection reason node {node_id or '?'} must return candidate")
            if op in BOOLEAN_OPS and return_kind != "boolean":
                errors.append(f"boolean reason node {node_id or '?'} must return boolean")
        else:
            errors.append(f"node {node_id or '?'} has unsupported type={node_type}")

        try:
            current = int(node_id)
        except ValueError:
            current = -1
        for parent_id in input_ids:
            if parent_id not in by_id:
                errors.append(f"node {node_id or '?'} references missing node #{parent_id}#")
                continue
            if int(parent_id) >= current:
                errors.append(f"node {node_id or '?'} references current/future node #{parent_id}#")
                continue
            children[parent_id].add(node_id)

    sink = str(attrs.get("sink", ""))
    sink_ids = _refs(sink)
    sink_id = sink_ids[0] if len(sink_ids) == 1 else ""
    if not sink_id or sink_id not in by_id:
        errors.append("plan sink must reference exactly one existing node")
    else:
        if children.get(sink_id):
            errors.append("declared sink is not terminal")
        terminals = {node_id for node_id, node_children in children.items() if not node_children}
        if terminals != {sink_id}:
            errors.append(f"plan must have one terminal sink; found {sorted(terminals)}")
    if not attrs.get("answer_type"):
        errors.append("plan has no answer_type")

    reason_nodes = [node for node in nodes if node.get("type") == "reason"]
    comparison_dimension = _comparison_dimension(question)
    comparison_words = (
        "both", "closer", "earlier", "fewer", "first", "later", "longer",
        "more", "older", "same", "shorter", "younger",
    )
    if any(word in str(question).lower() for word in comparison_words) and not reason_nodes:
        errors.append("comparison question requires an explicit Reason node")
    for reason in reason_nodes:
        op = _normalize_token(reason.get("op", ""))
        if op not in SELECTION_OPS | BOOLEAN_OPS:
            continue
        parents = [by_id.get(str(parent_id), {}) for parent_id in reason.get("input_ids") or []]
        if any(parent.get("type") != "search" for parent in parents):
            errors.append(
                f"typed comparison reason node {reason.get('id')} must consume Search sources"
            )
            continue
        relations = {
            _normalize_token(parent.get("relation", ""))
            for parent in parents
            if parent.get("relation")
        }
        if len(relations) != 1:
            errors.append(
                f"typed comparison reason node {reason.get('id')} mixes source relations: "
                f"{sorted(relations)}"
            )
        if comparison_dimension is not None:
            dimension, _ = comparison_dimension
            mismatched = [
                str(parent.get("id", "?"))
                for parent in parents
                if not _relation_matches_dimension(
                    str(parent.get("relation", "")), dimension
                )
            ]
            if mismatched:
                errors.append(
                    f"typed comparison reason node {reason.get('id')} uses source relation(s) "
                    f"outside the question's {dimension} dimension: {mismatched}"
                )
        if op in SELECTION_OPS:
            for parent in parents:
                parent_id = parent.get("id", "?")
                if not parent.get("candidate"):
                    errors.append(
                        f"selection source #{parent_id}# must preserve its answer candidate"
                    )
    return list(dict.fromkeys(errors))


def compile_search_query(
    node: Mapping[str, Any],
    step_results: Mapping[str, Any],
    reason_bindings: Mapping[str, Mapping[str, Any]],
) -> tuple[str, dict[str, Any]]:
    subject_spec = str(node.get("subject", "")).strip()
    refs = _refs(subject_spec)
    binding: dict[str, Any] = {}
    if refs:
        parent_id = refs[0]
        if parent_id in reason_bindings:
            binding = dict(reason_bindings[parent_id])
            subject = str(binding.get("selected_candidate") or binding.get("selected_subject") or "")
        else:
            subject = str(step_results.get(parent_id, ""))
    else:
        subject = subject_spec
    relation = str(node.get("relation", "")).replace("_", " ").strip()
    if not subject or is_invalid_conclusion(subject):
        return "", binding
    return " ".join(f"{subject} {relation}".split()), binding


def contract_reason_sources(
    node: Mapping[str, Any],
    plan_nodes: Sequence[Mapping[str, Any]],
    step_results: Mapping[str, Any],
) -> list[dict[str, Any]]:
    by_id = {str(item.get("id", "")): item for item in plan_nodes}
    sources = []
    for parent_id in node.get("input_ids") or []:
        parent = by_id.get(str(parent_id), {})
        sources.append({
            "id": str(parent_id),
            "task": str(parent.get("task", "")),
            "subject": str(parent.get("subject", "")),
            "relation": str(parent.get("relation", "")),
            "candidate": str(parent.get("candidate", "")),
            "value": str(step_results.get(str(parent_id), "")),
        })
    return sources


def make_contract_reason_binding(
    selected_ref: str,
    sources: Sequence[Mapping[str, Any]],
) -> dict[str, Any] | None:
    """Bind a model selection to one executed parent without copying model text."""
    selected = next(
        (source for source in sources if str(source.get("id", "")) == str(selected_ref)),
        None,
    )
    if selected is None:
        return None
    return {
        "kind": "contract_source_selection",
        "selected_ref": str(selected.get("id", "")),
        "selected_subject": str(selected.get("subject", "")),
        "selected_task": str(selected.get("task", "")),
        "selected_candidate": str(selected.get("candidate", "")),
        "selected_value": str(selected.get("value", "")),
        "selected_relation": str(selected.get("relation", "")),
    }


def _scalar(value: str) -> float | None:
    text = str(value or "").replace(",", "")
    years = re.findall(r"(?<!\d)(-?\d{4})(?!\d)", text)
    if years:
        return float(years[0])
    numbers = re.findall(r"(?<!\w)-?\d+(?:\.\d+)?", text)
    return float(numbers[0]) if numbers else None


def execute_deterministic_reason(
    node: Mapping[str, Any],
    sources: Sequence[Mapping[str, Any]],
) -> tuple[str, dict[str, Any] | None, str]:
    if not sources or any(is_invalid_conclusion(str(source.get("value", ""))) for source in sources):
        return "", None, "missing_source_value"
    op = _normalize_token(node.get("op", ""))
    return_kind = _normalize_token(node.get("return", ""))
    selected: Mapping[str, Any] | None = None
    if op in SELECTION_OPS:
        values = [_scalar(str(source.get("value", ""))) for source in sources]
        if any(value is None for value in values):
            return "", None, "non_scalar_source"
        numeric = [float(value) for value in values if value is not None]
        if op == "closest":
            anchor = _scalar(str(node.get("anchor", "")))
            if anchor is None:
                return "", None, "missing_anchor"
            index = min(range(len(numeric)), key=lambda i: abs(numeric[i] - anchor))
        elif op in {"later", "newer", "younger", "longer", "argmax"}:
            index = max(range(len(numeric)), key=numeric.__getitem__)
        else:
            index = min(range(len(numeric)), key=numeric.__getitem__)
        if numeric.count(numeric[index]) > 1:
            return "", None, "tied_source_values"
        selected = sources[index]
    elif op in BOOLEAN_OPS:
        normalized = [re.sub(r"\W+", " ", str(source.get("value", "")).lower()).strip() for source in sources]
        is_equal = len(set(normalized)) == 1
        answer = "Yes" if (is_equal == (op == "equal")) else "No"
        return answer, None, "deterministic_boolean"
    else:
        return "", None, "unsupported_deterministic_op"

    binding = {
        "kind": "contract_source_selection",
        "selected_ref": str(selected.get("id", "")),
        "selected_subject": str(selected.get("subject", "")),
        "selected_task": str(selected.get("task", "")),
        "selected_candidate": str(selected.get("candidate", "")),
        "selected_value": str(selected.get("value", "")),
        "selected_relation": str(selected.get("relation", "")),
    }
    if return_kind == "candidate":
        answer = binding["selected_candidate"]
    else:
        answer = binding["selected_value"]
    return str(answer), binding, "deterministic_selection"


def contract_protocol_metadata(plan: Mapping[str, Any], errors: Iterable[str]) -> dict[str, Any]:
    return {
        "version": CONTRACT_PROTOCOL_VERSION,
        "plan_attrs": dict(plan.get("attrs") or {}),
        "plan_nodes": list(plan.get("nodes") or []),
        "validation_errors": list(errors),
    }
