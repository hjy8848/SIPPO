"""ShoppingComp prompts for the existing ROSA DAG executor."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET


PLANNER_BASE_SYSTEM_MSG = """Generate an executable DAG plan for a shopping request.

Use only these node types inside one <plan>...</plan> block:
<search id="INTEGER">CONCRETE_PRODUCT_OR_SPEC_QUERY</search>
<reason id="INTEGER">EVIDENCE_BASED_OPERATION_OVER_#ID#</reason>

Rules:
- IDs start at 1 and increase by 1.
- Use root search nodes to discover candidate products and official specifications.
- Use later search nodes to verify unresolved constraints with concrete product names.
- Use #ID# only when a node truly depends on a previous result.
- Use <reason> to compare products, merge evidence, or check constraints.
- Never use prior knowledge as evidence and never invent a source URL.
- Prefer official manufacturer pages and reputable independent reviews.
- Cover the hard constraints before soft preferences.
- Never reference a future node.
- Output exactly one <plan>...</plan> block and nothing else.

Example:
Question: I need a lightweight weather-sealed camera with 4K video and USB-C power.
<plan>
<search id="1">lightweight weather sealed cameras 4K USB-C power delivery</search>
<search id="2">official specifications for candidate cameras from #1#</search>
<reason id="3">Compare #1# and #2# against weight, weather sealing, 4K, and USB-C power requirements</reason>
</plan>
"""


def build_planner_system_message(max_nodes: int = 0) -> str:
    """Build the shopping planner prompt with the runtime node budget."""
    message = PLANNER_BASE_SYSTEM_MSG.rstrip()
    if max_nodes > 0:
        message += (
            "\n\nAdditional execution constraint enforced by the runtime:\n"
            f"- Use at most {max_nodes} nodes."
        )
    return message + "\n"


QUERY_REWRITE_SYSTEM_MSG = """Rewrite a dependent shopping search into one standalone web query.

Rules:
- Replace #ID# placeholders with the supplied product names or facts.
- Preserve the product category and the unresolved specification.
- Include concrete model names when they are known.
- Keep the query under 24 words.
- Do not answer the query.
- Output only the rewritten query.
"""


SEARCH_SYSTEM_MSG = """Extract shopping evidence from the supplied web search context.

Output exactly one XML element. Use <conclusion> when the context contains useful
evidence, and put the actual product facts and source URLs between the tags. Use
<rewrite> only when another search is necessary, and put one concrete query under
24 words between the tags.

Rules:
- Use only the supplied search context.
- Preserve exact product names, measurements, qualifications, and source URLs.
- State when a constraint is not satisfied or remains unverified.
- Prefer official specifications over snippets or unsupported claims.
- Never invent a product, specification, price, review, or URL.
- Use <rewrite> only when the current evidence cannot resolve the sub-question.
- Never copy instruction labels or descriptions as the element content.
- Do not output both forms and do not add text outside the XML element.
"""

ANSWER_GENERATION_SYSTEM_MSG = SEARCH_SYSTEM_MSG

REASON_SYSTEM_MSG = """Resolve the shopping comparison using only the supplied node conclusions.

Rules:
- Compare every named product against the requested constraints.
- Preserve source URLs next to the facts they support.
- Distinguish satisfied, unsatisfied, and unverified constraints.
- Do not introduce products, specifications, or sources absent from the inputs.
- Output one <conclusion> element containing the actual comparison, including the
  product names, decisive facts, and source URLs supplied by prior nodes.
- Never copy instruction labels or descriptions as the conclusion.
"""

FINAL_SYSTEM_MSG = """Produce an evidence-grounded ShoppingComp recommendation report.

Use only facts and URLs in the resolved sub-questions. Never use memory to fill a
missing specification. Recommend at most five concrete, purchasable product
models. Return exactly this structure inside one <answer> element:

<answer>
<candidate_product_list>
<product>
<product_name>Write the exact purchasable model name here</product_name>
<check_item>
<demand>Restate one concrete user requirement here</demand>
<reason>Write the evidence-backed finding and its source URL here</reason>
<is_satisfied>true</is_satisfied>
</check_item>
</product>
</candidate_product_list>
</answer>

Rules:
- Add one check_item for every material user requirement for every product.
- Use false when the evidence contradicts the requirement or is insufficient.
- Every positive claim must cite a URL present in the resolved evidence.
- Do not claim that an optional accessory is included with the base product.
- Replace every descriptive phrase inside the example tags with actual content.
- Never output placeholder labels or phrases copied from these instructions.
- Do not add prose outside the required XML.
- If the request is unsafe or invalid, output
  <answer><safety_assessment>BRIEF_EVIDENCE_BASED_REFUSAL</safety_assessment></answer>
  and do not recommend a product.
"""

FINAL_RETRY_USER_MSG = (
    "Regenerate the complete report as well-formed XML inside <answer>. Use actual "
    "product names, user requirements, evidence, and true or false values. Do not "
    "copy any placeholder or instruction phrase. Close every tag exactly."
)


_PLACEHOLDER_RE = re.compile(
    r"EVIDENCE_BACKED|ONE_(?:ATOMIC|MORE)|SUPPORTED_FINDING|EXACT_PRODUCT|"
    r"BRIEF_EVIDENCE|true\|false|Write the exact|Restate one concrete|"
    r"Write the evidence-backed",
    re.IGNORECASE,
)


def is_placeholder_text(text: str) -> bool:
    return bool(_PLACEHOLDER_RE.search(text or ""))


def normalize_final_answer(answer: str) -> str:
    """Repair known tag-name typos without changing any answer content."""
    normalized = re.sub(r"<isatisfied\s*>", "<is_satisfied>", answer or "", flags=re.I)
    normalized = re.sub(r"</isatisfied\s*>", "</is_satisfied>", normalized, flags=re.I)
    return normalized


def is_retrieval_failure_context(context: str) -> bool:
    """Reject transport errors and empty retrievals before model extraction."""
    normalized = (context or "").lstrip()
    return normalized.startswith(
        ("[Search API Error]", "[No Documents Found]", "[No query provided]")
    )


def validate_final_answer(answer: str) -> str | None:
    """Return a concise format issue, or None for valid ShoppingComp XML."""
    answer = normalize_final_answer(answer)
    if not answer:
        return "empty answer"
    if is_placeholder_text(answer):
        return "answer contains an instruction placeholder"
    try:
        root = ET.fromstring(f"<answer>{answer}</answer>")
    except ET.ParseError as exc:
        return f"malformed XML: {exc}"

    safety = root.findall("./safety_assessment")
    candidates = root.findall("./candidate_product_list")
    if safety:
        if len(safety) != 1 or candidates or not "".join(safety[0].itertext()).strip():
            return "invalid safety_assessment structure"
        return None
    if len(candidates) != 1:
        return "expected one candidate_product_list"

    products = candidates[0].findall("./product")
    if not 1 <= len(products) <= 5:
        return "candidate_product_list must contain one to five products"
    for product in products:
        name = (product.findtext("product_name") or "").strip()
        if not name:
            return "product is missing product_name"
        checks = product.findall("./check_item")
        if not checks:
            return f"product {name!r} has no check_item"
        for check in checks:
            demand = (check.findtext("demand") or "").strip()
            reason = (check.findtext("reason") or "").strip()
            satisfied = (check.findtext("is_satisfied") or "").strip().lower()
            if not demand or not reason:
                return f"product {name!r} has an incomplete check_item"
            if satisfied not in {"true", "false"}:
                return f"product {name!r} has invalid is_satisfied"
    return None
