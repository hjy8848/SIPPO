"""Shared ROSA prompts and plan/output validation."""

import re
import string
from typing import Any, Iterable, Sequence


REFERENCE_RE = re.compile(r"#(\d+)#")
WORD_RE = re.compile(r"[^\W_]+", re.UNICODE)
SELECTION_RE = re.compile(
    r'<selection\s+ref=["\']#?(\d+)#?["\']\s*>\s*'
    r'<value>(.*?)</value>\s*</selection>',
    re.IGNORECASE | re.DOTALL,
)

DEGENERATE_EXACT_TEXT = {
    "one atomic retrieval question",
    "logic over previous #id# results",
    "logic over previous results",
    "if the answer is present in the context",
    "if the answer is present in context",
    "exact answer",
    "supported short answer",
    "derived short answer",
    "supported final answer",
    "concrete rewritten search query",
    "concrete search query",
    "search query",
    "retrieval query",
    "better standalone retrieval query",
}

QUESTION_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "between", "both", "by",
    "did", "do", "does", "earlier", "first", "for", "from", "has", "have",
    "how", "in", "is", "later", "longer", "more", "of", "on", "or", "than",
    "that", "the", "their", "to", "was", "were", "what", "when", "where",
    "which", "who", "whose", "with", "younger", "older",
}


def find_reference_ids(text: str) -> list[str]:
    """Return canonical ``#ID#`` references."""
    return REFERENCE_RE.findall(str(text))


def has_reference(text: str) -> bool:
    return REFERENCE_RE.search(str(text)) is not None


def _normalized_text(text: str) -> str:
    text = str(text or "").strip().lower()
    text = re.sub(r"[_{}<>]+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation.replace("#", "")))
    return " ".join(text.split())


def is_degenerate_text(text: str) -> bool:
    """Reject prompt metavariables and punctuation-only pseudo answers."""
    raw_text = str(text or "").strip()
    if re.fullmatch(r"\{[A-Z0-9_#]+\}", raw_text):
        return True
    normalized = _normalized_text(text)
    if not normalized or normalized in DEGENERATE_EXACT_TEXT:
        return True
    if normalized.replace("#", "").strip() in {"", "x", "id", "integer"}:
        return True
    if not any(ch.isalnum() for ch in normalized):
        return True
    return False


def is_invalid_conclusion(conclusion: str) -> bool:
    """Return whether a search/reason conclusion is unusable evidence."""
    if is_degenerate_text(conclusion):
        return True
    normalized = _normalized_text(conclusion)
    bad_phrases = (
        "insufficient information",
        "no information",
        "not enough information",
        "not provided",
        "not mentioned",
        "cannot be determined",
        "can't be determined",
        "cant be determined",
        "unknown",
        "no conclusion",
    )
    return any(phrase in normalized for phrase in bad_phrases)


def _content_words(text: str) -> set[str]:
    return {
        word.lower()
        for word in WORD_RE.findall(str(text or ""))
        if len(word) >= 3 and word.lower() not in QUESTION_STOPWORDS
    }


def validate_plan_semantics(question: str, nodes: Iterable[Any]) -> list[str]:
    """Reject copied schemas and root searches unrelated to the question."""
    nodes = list(nodes)
    errors: list[str] = []
    question_words = _content_words(question)
    search_count = 0

    for node in nodes:
        node_type, node_id, text = _node_parts(node)
        text = text.strip()
        if is_degenerate_text(text):
            errors.append(f"Node {node_id or '?'} contains placeholder or empty text.")
            continue
        if node_type == "search":
            search_count += 1
            if not has_reference(text) and question_words and not (_content_words(text) & question_words):
                errors.append(f"Root search node {node_id or '?'} is not grounded in the question.")
        elif node_type == "reason" and not has_reference(text):
            errors.append(f"Reason node {node_id or '?'} does not reference previous evidence.")

    if search_count == 0:
        errors.append("Plan must contain at least one search node.")
    return errors


def is_supported_answer(answer: str, conclusions: Iterable[str]) -> bool:
    """Check whether a final short answer is stated by a valid node conclusion."""
    answer_norm = _normalized_text(answer)
    if is_degenerate_text(answer) or answer_norm in {"insufficient information", "unknown"}:
        return False
    for conclusion in conclusions:
        if is_invalid_conclusion(conclusion):
            continue
        conclusion_norm = _normalized_text(conclusion)
        if answer_norm == conclusion_norm:
            return True
        if len(answer_norm) >= 4 and re.search(
            rf"(?<!\w){re.escape(answer_norm)}(?!\w)", conclusion_norm
        ):
            return True
    return False

COMMON_PLANNER_RULES = """Generate a DAG execution plan for the question.

Use only these node types inside one <plan>...</plan> block:
<search id="INTEGER">{CONCRETE_QUESTION}</search>
<reason id="INTEGER">{CONCRETE_OPERATION_OVER_#ID#}</reason>

The braced uppercase strings above are grammar metavariables. Replace every
metavariable with question-specific content; never copy a metavariable or an
instruction phrase into the plan.

Rules:
- IDs start at 1 and increase by 1.
- Ask one fact per <search>.
- Use #ID# only when a node truly depends on a previous result.
- Use <reason> for comparison, arithmetic, extraction, filtering, merging, or final synthesis.
- Use <reason> only when all required facts are already available in previous conclusions. If a new external fact is needed, use <search>, not <reason>.
- Never reference a future node.
- Output exactly one <plan>...</plan> block and nothing else.
"""

PLANNER_BASE_SYSTEM_MSG = COMMON_PLANNER_RULES + """

Example:
Question: Who is the father of Ada Lovelace's husband?
<plan>
<search id="1">Who was Ada Lovelace's husband?</search>
<search id="2">Who was the father of #1#?</search>
</plan>
"""

def build_planner_system_message(max_nodes: int = 5) -> str:
    """Build a planner prompt that matches the runtime node-limit semantics."""
    message = PLANNER_BASE_SYSTEM_MSG.rstrip()
    if max_nodes > 0:
        message += (
            "\n\nAdditional execution constraints enforced by the runtime:\n"
            f"- Use at most {max_nodes} nodes."
        )
    return message + "\n"


PLANNER_SYSTEM_MSG = build_planner_system_message()

QUERY_REWRITE_SYSTEM_MSG = """Rewrite a dependent search query into a standalone retrieval query.

Rules:
- Replace #ID# placeholders with their provided values.
- Keep important constraints from the original goal, such as dates, locations, and entity names.
- Make the query concise and searchable.
- Do not answer the query.
- Output only the rewritten query, with no tags or explanation.
"""

SEARCH_SYSTEM_MSG = """Answer the current sub-question using only the search context.

Output exactly one of these XML forms:
<conclusion>{SUPPORTED_SHORT_ANSWER}</conclusion>
<rewrite>{QUESTION_SPECIFIC_SEARCH_QUERY}</rewrite>

The braced uppercase strings are metavariables, not valid outputs. Replace
them with text grounded in the current sub-question and search context.

Rules:
- The <rewrite> must be a real search query containing the key entity names from the current sub-question.
- Never output generic phrases such as "better standalone retrieval query", "search query", or "retrieval query".
- Do not explain.
- Do not output both <conclusion> and <rewrite>.
- Use <conclusion> only when the answer is directly supported by the context.
- Never copy wording from these instructions as a conclusion or rewrite.
"""

ANSWER_GENERATION_SYSTEM_MSG = SEARCH_SYSTEM_MSG

REASON_SYSTEM_MSG = """Resolve the logic task.

Rules:
- Use only the information stated in the task.
- Perform the required comparison, computation, extraction, filtering, or synthesis.
- Output only <conclusion>{DERIVED_SHORT_ANSWER}</conclusion>.
- Replace the braced metavariable; an ellipsis or instruction phrase is invalid.
"""

TYPED_REASON_SYSTEM_MSG = """Resolve a selection task over named evidence sources.

Output exactly this XML form:
<selection ref="#ID#"><value>{EXACT_SOURCE_VALUE}</value></selection>

Rules:
- Choose exactly one source listed in Available Sources.
- The ref must copy that source's #ID#.
- The value must exactly copy that same source's value.
- Decide from the logic task; do not select by source order.
- Do not output a bare value, a conclusion tag, an explanation, or any other text.
"""

LOCKED_QUERY_REWRITE_SYSTEM_MSG = """Compile a dependent task into a standalone search query.

The Selected Source is an immutable provenance binding. Use the entity described
by that source and do not substitute an entity from any Rejected Source.

Rules:
- Preserve the downstream relation or attribute requested by the Draft.
- Mention identifying terms from the Selected Source.
- Never mention identifying terms unique to a Rejected Source.
- Do not answer the query.
- Output only the rewritten query, with no tags or explanation.
"""

FINAL_SYSTEM_MSG = """Synthesize the resolved sub-questions to answer the original question.

Rules:
- Output only <answer>{SUPPORTED_FINAL_ANSWER}</answer>.
- Replace the braced metavariable; an ellipsis or instruction phrase is invalid.
- Keep the answer concise.
- If the answer is a name, date, location, number, or yes/no, output only that value.
- Use only facts in the resolved sub-questions. Do not answer from memory when
  the resolved evidence is missing or contradictory.
- Do not add explanations.
"""


def build_planning_messages(question: str, max_nodes: int = 5) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": build_planner_system_message(max_nodes)},
        {"role": "user", "content": f"Question: {question}\nGenerate the execution plan."},
    ]


def build_query_rewrite_messages(
    question: str,
    raw_query: str,
    step_results: dict[str, Any] | dict[int, Any] | None = None,
) -> list[dict[str, str]]:
    references = find_reference_ids(raw_query)
    if references:
        values = step_results or {}
        lines = [f"Original Goal: {question}", f"Draft: {raw_query}"]
        for reference in references:
            value = values.get(reference, values.get(int(reference), "UNKNOWN"))
            lines.append(f"# {reference} = {value}")
        lines.append("Rewritten query:")
        user_content = "\n".join(lines)
    else:
        user_content = raw_query
    return [
        {"role": "system", "content": QUERY_REWRITE_SYSTEM_MSG},
        {"role": "user", "content": user_content},
    ]


def build_answer_generation_messages(query: str, documents: Sequence[str]) -> list[dict[str, str]]:
    context = "\n".join(documents)
    return [
        {"role": "system", "content": ANSWER_GENERATION_SYSTEM_MSG},
        {
            "role": "user",
            "content": f"**Current Sub-question**: {query}\n\n**Search Context**:\n{context}",
        },
    ]


def build_reasoner_messages(logic_task: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": REASON_SYSTEM_MSG},
        {"role": "user", "content": f"**Current Logic Task**: {logic_task}"},
    ]


def parse_typed_reason_selection(text: str) -> dict[str, str] | None:
    """Parse the strict source-selection result used by typed handoff."""
    match = SELECTION_RE.search(str(text or ""))
    if not match:
        return None
    return {"ref": match.group(1), "value": match.group(2).strip()}


def build_typed_reason_messages(
    question: str,
    logic_task: str,
    sources: Sequence[dict[str, Any]],
) -> list[dict[str, str]]:
    lines = [
        f"Original Question: {question}",
        f"Logic Task: {logic_task}",
        "Available Sources:",
    ]
    for source in sources:
        lines.append(
            f"#{source['id']}# | task={source['task']} | value={source['value']}"
        )
    return [
        {"role": "system", "content": TYPED_REASON_SYSTEM_MSG},
        {"role": "user", "content": "\n".join(lines)},
    ]


def validate_typed_reason_selection(
    selection: dict[str, str] | None,
    sources: Sequence[dict[str, Any]],
) -> str | None:
    """Return a validation error, or ``None`` for a provenance-safe selection."""
    if selection is None:
        return "missing typed <selection> output"
    by_id = {str(source["id"]): source for source in sources}
    source = by_id.get(str(selection.get("ref", "")))
    if source is None:
        return "selection ref is not a declared parent source"
    if is_invalid_conclusion(str(source.get("value", ""))):
        return "selected source has no usable value"
    if _normalized_text(selection.get("value", "")) != _normalized_text(source["value"]):
        return "selection value does not match the referenced source"
    return None


def make_typed_reason_binding(
    selection: dict[str, str],
    sources: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    by_id = {str(source["id"]): dict(source) for source in sources}
    selected = by_id[str(selection["ref"])]
    rejected = [source for source_id, source in by_id.items() if source_id != str(selection["ref"])]
    return {
        "kind": "source_selection",
        "selected_ref": str(selection["ref"]),
        "selected_task": str(selected["task"]),
        "selected_value": str(selected["value"]),
        "rejected_sources": rejected,
    }


def build_locked_query_rewrite_messages(
    question: str,
    raw_query: str,
    binding: dict[str, Any],
) -> list[dict[str, str]]:
    lines = [
        f"Original Goal: {question}",
        f"Draft: {raw_query}",
        f"Selected Source: #{binding['selected_ref']}#",
        f"Selected Source Task: {binding['selected_task']}",
        f"Selected Source Value: {binding['selected_value']}",
    ]
    for source in binding.get("rejected_sources", []):
        lines.append(
            f"Rejected Source: #{source['id']}# | task={source['task']} | value={source['value']}"
        )
    lines.append("Standalone query:")
    return [
        {"role": "system", "content": LOCKED_QUERY_REWRITE_SYSTEM_MSG},
        {"role": "user", "content": "\n".join(lines)},
    ]


def _distinctive_words(task: str, other_tasks: Sequence[str]) -> set[str]:
    words = _content_words(task)
    other_words: set[str] = set()
    for other_task in other_tasks:
        other_words.update(_content_words(other_task))
    return words - other_words


def validate_locked_query_rewrite(query: str, binding: dict[str, Any]) -> str | None:
    """Reject rewrites that drop the selected source or switch to a sibling."""
    if is_degenerate_text(query):
        return "empty or degenerate rewrite"
    query_words = _content_words(query)
    rejected = list(binding.get("rejected_sources", []))
    rejected_tasks = [str(source.get("task", "")) for source in rejected]
    selected_distinctive = _distinctive_words(
        str(binding["selected_task"]), rejected_tasks
    )
    for source in rejected:
        sibling_distinctive = _distinctive_words(
            str(source.get("task", "")), [str(binding["selected_task"])]
        )
        if query_words & sibling_distinctive:
            return f"rewrite mentioned identifying terms from rejected source #{source['id']}#"
    if selected_distinctive and not (query_words & selected_distinctive):
        return "rewrite dropped identifying terms from the selected source"
    return None


def build_provenance_preserving_query(raw_query: str, binding: dict[str, Any]) -> str:
    """Build a deterministic fallback that cannot switch the selected source."""
    provenance = (
        f'the entity from source task "{binding["selected_task"]}" '
        f'(observed value: {binding["selected_value"]})'
    )
    query = str(raw_query)
    reason_node_id = str(binding.get("reason_node_id", ""))
    if reason_node_id:
        query = query.replace(f"#{reason_node_id}#", provenance)
    else:
        query = re.sub(r"#\d+#", provenance, query, count=1)
    if query == str(raw_query):
        query = f"{raw_query}; selected source: {provenance}"
    return " ".join(query.split())


def build_summarization_messages(
    question: str,
    subquestions: Sequence[str],
    conclusions: Sequence[str],
) -> list[dict[str, str]]:
    if len(subquestions) != len(conclusions):
        raise ValueError("subquestions and conclusions must have equal length")
    history = "\n\n".join(
        f"Task {index}: {subquestion}\nConclusion: {conclusion}"
        for index, (subquestion, conclusion) in enumerate(zip(subquestions, conclusions), start=1)
    )
    return [
        {"role": "system", "content": FINAL_SYSTEM_MSG},
        {
            "role": "user",
            "content": f"**Original Question**: {question}\n\n**Resolved Sub-questions**:\n{history}",
        },
    ]


def _node_parts(node: Any) -> tuple[str, str, str]:
    if isinstance(node, dict):
        node_type = str(node.get("type", ""))
        node_id = str(node.get("id", ""))
        text = str(node.get("query", node.get("task", "")))
        return node_type, node_id, text
    if isinstance(node, (tuple, list)) and len(node) >= 3:
        return str(node[0]), str(node[1]), str(node[2])
    return "", "", ""


def validate_plan_structure(nodes: Iterable[Any], max_nodes: int = 0) -> list[str]:
    """Return generic DAG validation errors without imposing a topology label."""
    nodes = list(nodes)
    if max_nodes > 0 and len(nodes) > max_nodes:
        return [f"Plan has {len(nodes)} nodes; maximum is {max_nodes}."]
    return []
