"""ShoppingBench prompts and output validation for the ROSA DAG executor."""

from __future__ import annotations

import json
import itertools
import re
import xml.etree.ElementTree as ET
from html import escape, unescape
from typing import Any, Mapping, Sequence
from urllib.parse import parse_qsl, unquote_plus


PLANNER_BASE_SYSTEM_MSG = """Generate an executable DAG plan for a local ShoppingBench request.

Use only these node types inside one <plan>...</plan> block:
<search id="INTEGER">SEARCH_QUERY_OR_JSON_FILTER</search>
<reason id="INTEGER">EVIDENCE_BASED_OPERATION_OVER_#ID#</reason>

The local catalog search accepts either plain text or a JSON object with these fields:
q, page, shop_id, price, sort, service, product_ids. Price requires two numeric
bounds joined by a hyphen; keep one-sided price constraints in the q text;
sort is order, priceasc, or pricedesc; service values are official,
freeShipping, COD, and flashsale. Use product_ids to inspect known candidates.

Rules:
- IDs start at 1 and increase by 1.
- For a request asking for one product, use exactly one plain-text search node
  copied closely from the request and do not add a reason node.
- Do not add a price, service, shop, sort, or SKU filter that is absent from the request.
- Keep one-sided price constraints in plain search text; never convert them to a range.
- Search separately for distinct products only when the request asks for several.
- Preserve product IDs in every comparison and final decision.
- Use filters only when the request explicitly requires them.
- For shop tasks, verify that every selected product has the same shop ID.
- For voucher tasks, retrieve prices and calculate the final discounted total.
- Use #ID# only when a node truly depends on a previous result.
- Never reference a future node or invent catalog facts.
- Output exactly one <plan>...</plan> block and nothing else.

Example:
Question: Find a 12 inch heavy-duty rubber wheel for a pushcart.
<plan>
<search id="1">12 inch heavy-duty rubber wheel for a pushcart</search>
</plan>
"""


def build_planner_system_message(max_nodes: int = 0) -> str:
    message = PLANNER_BASE_SYSTEM_MSG.rstrip()
    if max_nodes > 0:
        message += f"\n\nAdditional runtime constraint:\n- Use at most {max_nodes} nodes."
    return message + "\n"


def build_contract_planner_system_message(max_nodes: int = 0) -> str:
    """Build the ShoppingBench specialization of the shared CP-DAGENT grammar."""
    limit = f"- Use at most {max_nodes} nodes.\n" if max_nodes > 0 else ""
    return f"""Generate a contract-preserving Search/Reason DAG for a local ShoppingBench request.

Output exactly one XML plan with this grammar:
<plan sink="#ID#" answer_type="product_ids">
<search id="ID" subject="CATALOG_OR_#ID#" relation="catalog_candidates|product_details">PLAIN_QUERY_OR_JSON_FILTER</search>
<reason id="ID" op="filter_constraints|intersect_product|join_shop|apply_voucher|argmin|argmax" inputs="#ID#,#ID#" return="product_ids">CONCRETE_SELECTION_TASK</reason>
</plan>

The catalog accepts plain search text or one compact JSON object containing only:
q, page, shop_id, price, sort, service, product_ids. Price is a two-sided range
joined by a hyphen. Keep one-sided price constraints in q. Valid sort values are
order, priceasc, and pricedesc. Valid service values are official, freeShipping,
COD, and flashsale.

Contract:
- IDs start at 1, increase by 1, and may reference only earlier nodes.
{limit}- Every node must contribute to the single declared sink.
- Preserve all explicit product, SKU, price, service, shop, and voucher constraints.
- Do not invent a filter absent from the request.
- Use separate root Search nodes for distinct requested products.
- Use filter_constraints for one retrieved set, intersect_product for repeated
  searches for the same product, join_shop for distinct products that must come
  from one shop, and apply_voucher for a budget/voucher request.
- A dependent Search may put exactly one prior reference in subject; the runtime will
  bind product IDs from that parent without asking QueryRewrite to reinterpret its text.
- Never place an example or guessed Product ID in the plan.
- Do not declare an out field. Runtime artifacts carry Product IDs and provenance.
- Output exactly one plan and no prose. Do not copy an example query or add a
  second Search for a constraint that is absent from the request.
"""


_CONTRACT_QUERY_FIELDS = {
    "q",
    "page",
    "shop_id",
    "price",
    "sort",
    "service",
    "product_ids",
}
_PRODUCT_EVIDENCE_ID_RE = re.compile(r"\bProduct ID:\s*(\d+)\b", re.I)
_DOCUMENT_MARKER_RE = re.compile(r"(?m)^\[1_(\d+)\]:\s*")
_FIELD_RE = re.compile(
    r"(?m)^(Product ID|Shop ID|Title|Price|Services|Sold Count|Short Description|"
    r"Description|SKU Options|Attributes):\s*(.*)$"
)
_ANSWER_PRODUCT_ID_RE = re.compile(
    r"(?:Product\s+ID\s*:\s*|\bid\s*=\s*[\"']?)(\d+)", re.I
)
_SEARCH_BODY_RE = re.compile(r"<search\b[^>]*>(.*?)</search>", re.I | re.S)
_QUERY_TAG_RE = re.compile(r"<q>(.*?)</q>", re.I | re.S)
_QUERY_LINE_RE = re.compile(r"^\s*q\s*[:=]\s*(.*?)\s*$", re.I)
_TRAILING_QUERY_FIELD_RE = re.compile(
    r"(?:\s+|&)(?:page|shop_id|price|sort|service|product_ids)\s*[:=].*$", re.I
)
_VISIBLE_PRODUCT_ITEM_RE = re.compile(
    r"\((\d+)\)\s*(.*?)(?=;\s*\(\d+\)\s|\.\s*(?:Purchase terms:|\?|$))",
    re.I,
)
_MERCHANT_RE = re.compile(r"\b(?:shop|store|seller)\b")
_SHARED_MERCHANT_RE = re.compile(
    r"\b(?:one|single|common|same)\s+(?:shop|store|seller)\b"
)
_COLLECTIVE_PURCHASE_MARKERS = (
    "all requested items",
    "all these",
    "several items",
    "these items",
    "these products",
    "this group",
    "this list",
    "following items",
    "everything",
    "sell all",
)


def classify_shopping_request(question: str) -> str:
    """Infer only the control-flow family from the user-visible request."""
    normalized = " ".join(str(question or "").lower().split())
    if "voucher" in normalized and any(
        marker in normalized for marker in ("budget", "discount", "purchase terms")
    ):
        return "voucher"
    if _SHARED_MERCHANT_RE.search(normalized):
        return "shop"
    if _MERCHANT_RE.search(normalized) and any(
        marker in normalized for marker in _COLLECTIVE_PURCHASE_MARKERS
    ):
        return "shop"
    return "product"


def _clean_query_candidate(value: str) -> str:
    value = unquote_plus(str(value or "")).strip().strip("\"'")
    value = _TRAILING_QUERY_FIELD_RE.sub("", value).strip().strip("\"'")
    return " ".join(value.split())


def extract_atomic_search_queries(raw_plan: str) -> list[str]:
    """Recover semantic Search proposals without trusting model control flow."""
    queries: list[str] = []
    for body in _SEARCH_BODY_RE.findall(str(raw_plan or "")):
        body = unescape(body)
        tagged = [_clean_query_candidate(value) for value in _QUERY_TAG_RE.findall(body)]
        if tagged:
            queries.extend(tagged)
            continue

        compact = " ".join(body.split())
        if "&q=" in compact.lower():
            parsed = [
                _clean_query_candidate(value)
                for key, value in parse_qsl(compact, keep_blank_values=False)
                if key.lower() == "q"
            ]
            if parsed:
                queries.extend(parsed)
                continue

        line_queries = []
        for line in body.splitlines():
            match = _QUERY_LINE_RE.match(line)
            if match:
                line_queries.append(_clean_query_candidate(match.group(1)))
        if line_queries:
            queries.extend(line_queries)
            continue

        plain = re.sub(r"<[^>]+>", " ", body)
        queries.append(_clean_query_candidate(plain))
    return list(dict.fromkeys(query for query in queries if query))


def extract_visible_product_queries(question: str) -> list[str]:
    """Split only explicitly numbered products from the user-visible request."""
    text = " ".join(str(question or "").split())
    if "Product details: " not in text:
        return []
    details = text.split("Product details: ", 1)[1]
    return [
        match.group(2).strip()
        for match in _VISIBLE_PRODUCT_ITEM_RE.finditer(details)
        if match.group(2).strip()
    ]


def compile_shoppingbench_plan(
    question: str,
    raw_plan: str,
    *,
    max_nodes: int = 5,
) -> tuple[str, dict[str, Any]]:
    """Compile model Search semantics into a system-owned executable DAG."""
    request_kind = classify_shopping_request(question)
    queries = extract_atomic_search_queries(raw_plan)
    minimum_queries = 2 if request_kind == "shop" else 1
    if len(queries) < minimum_queries:
        visible_queries = extract_visible_product_queries(question)
        if len(visible_queries) >= minimum_queries:
            queries = visible_queries
    if request_kind == "product":
        queries = queries[:1]
    if not queries:
        raise ValueError("planner output contains no atomic Search query")
    if request_kind == "shop" and len(queries) < 2:
        raise ValueError("shop request requires at least two atomic Search queries")

    reason_nodes = 0 if request_kind == "product" else 1
    max_searches = max_nodes - reason_nodes if max_nodes > 0 else len(queries)
    if max_searches <= 0:
        raise ValueError(
            f"compiled plan needs {len(queries) + reason_nodes} nodes; maximum is {max_nodes}"
        )
    if request_kind == "voucher":
        queries = queries[:max_searches]
    elif len(queries) > max_searches:
        raise ValueError(
            f"compiled plan needs {len(queries) + reason_nodes} nodes; maximum is {max_nodes}"
        )

    lines = []
    for index, query in enumerate(queries, start=1):
        lines.append(
            f'<search id="{index}" subject="catalog" '
            f'relation="catalog_candidates">{escape(query)}</search>'
        )
    sink_id = len(queries)
    if request_kind != "product":
        sink_id += 1
        inputs = ",".join(f"#{index}#" for index in range(1, len(queries) + 1))
        op = "join_shop" if request_kind == "shop" else "apply_voucher"
        task = (
            "Select one requested product from each input with one shared Shop ID."
            if request_kind == "shop"
            else "Apply the stated voucher and budget to the retrieved products."
        )
        lines.append(
            f'<reason id="{sink_id}" op="{op}" inputs="{inputs}" '
            f'return="product_ids">{task}</reason>'
        )
    plan = (
        f'<plan sink="#{sink_id}#" answer_type="product_ids">\n'
        + "\n".join(lines)
        + "\n</plan>"
    )
    metadata = {
        "compiler": "shoppingbench-system-topology-v2-voucher-multisearch",
        "request_kind": request_kind,
        "queries": queries,
        "model_controls": ["search_query_semantics"],
        "system_controls": ["node_ids", "dependencies", "operator", "sink"],
        "gold_used": False,
    }
    return plan, metadata


def _validated_root_query(task: str) -> str:
    task = " ".join(str(task or "").split()).strip()
    if not task or is_placeholder_text(task):
        return ""
    if not task.startswith("{"):
        return task
    try:
        payload = json.loads(task)
    except json.JSONDecodeError:
        return ""
    if not isinstance(payload, dict) or not payload or set(payload) - _CONTRACT_QUERY_FIELDS:
        return ""
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def compile_contract_search_query(
    node: Mapping[str, Any],
    step_results: Mapping[str, Any],
    reason_bindings: Mapping[str, Mapping[str, Any]],
) -> tuple[str, dict[str, Any]]:
    """Compile a ShoppingBench Search without a generative query-rewrite hop."""
    input_ids = [str(value) for value in node.get("input_ids") or []]
    if not input_ids:
        return _validated_root_query(str(node.get("task", ""))), {}

    parent_id = input_ids[0]
    binding = dict(reason_bindings.get(parent_id) or {})
    candidates = [
        str(binding.get("selected_candidate") or "").strip(),
        str(binding.get("selected_subject") or "").strip(),
    ]
    evidence = str(step_results.get(parent_id, ""))
    product_ids = [value for value in candidates if value.isdigit()]
    product_ids.extend(_PRODUCT_EVIDENCE_ID_RE.findall(evidence))
    product_ids = list(dict.fromkeys(product_ids))
    if not product_ids:
        return "", binding
    query = json.dumps(
        {"product_ids": ",".join(product_ids)},
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return query, binding


def parse_product_set(
    context: str,
    *,
    query: str = "",
    source_node: str = "",
) -> list[dict[str, Any]]:
    """Parse retriever-owned catalog rows into provenance-carrying artifacts."""
    text = str(context or "")
    markers = list(_DOCUMENT_MARKER_RE.finditer(text))
    products: list[dict[str, Any]] = []
    for index, marker in enumerate(markers):
        end = markers[index + 1].start() if index + 1 < len(markers) else len(text)
        block = text[marker.end() : end]
        fields: dict[str, str] = {}
        for match in _FIELD_RE.finditer(block):
            fields[match.group(1).lower().replace(" ", "_")] = match.group(2).strip()
        product_id_match = _PRODUCT_EVIDENCE_ID_RE.search(block)
        product_id = (
            product_id_match.group(1)
            if product_id_match is not None
            else fields.get("product_id", "")
        )
        if not product_id.isdigit():
            continue
        price: float | None = None
        try:
            price = float(fields.get("price", ""))
        except ValueError:
            pass
        services = [
            value.strip()
            for value in fields.get("services", "").split(",")
            if value.strip()
        ]
        products.append(
            {
                "product_id": product_id,
                "shop_id": fields.get("shop_id", ""),
                "title": fields.get("title", ""),
                "price": price,
                "services": services,
                "sku_options": fields.get("sku_options", ""),
                "attributes": fields.get("attributes", ""),
                "retrieval_rank": int(marker.group(1)),
                "source_node": str(source_node),
                "source_query": str(query),
            }
        )
    return products


def select_supported_products(
    products: Sequence[Mapping[str, Any]],
    model_conclusion: str,
) -> list[dict[str, Any]]:
    """Use model selection only as control; all returned data stays retriever-owned."""
    conclusion = str(model_conclusion or "")
    selected_ids = _ANSWER_PRODUCT_ID_RE.findall(conclusion)
    selected_ids.extend(re.findall(r"(?m)^\s*(\d+)\s*$", conclusion))
    selected_ids = list(dict.fromkeys(selected_ids))
    by_id = {str(product.get("product_id", "")): dict(product) for product in products}
    return [by_id[product_id] for product_id in selected_ids if product_id in by_id]


def annotate_product_selection(
    products: Sequence[Mapping[str, Any]],
    model_conclusion: str,
) -> list[dict[str, Any]]:
    selected_ids = {
        str(product.get("product_id", ""))
        for product in select_supported_products(products, model_conclusion)
    }
    return [
        {**dict(product), "model_selected": str(product.get("product_id", "")) in selected_ids}
        for product in products
    ]


def product_set_conclusion(products: Sequence[Mapping[str, Any]]) -> str:
    lines = []
    for product in products:
        price = product.get("price")
        price_text = "unknown" if price is None else str(price)
        lines.append(
            "Product ID: {product_id}; Shop ID: {shop_id}; Price: {price}; Title: {title}".format(
                product_id=product.get("product_id", ""),
                shop_id=product.get("shop_id", ""),
                price=price_text,
                title=product.get("title", ""),
            )
        )
    return "\n".join(lines) if lines else "Insufficient information"


def _deduplicate_products(products: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for product in products:
        product_id = str(product.get("product_id", ""))
        if product_id and product_id not in seen:
            seen.add(product_id)
            result.append(dict(product))
    return result


def _best_same_shop_group(
    source_sets: Sequence[Sequence[Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    if not source_sets or any(not source for source in source_sets):
        return []
    best: tuple[tuple[int, tuple[str, ...]], list[dict[str, Any]]] | None = None
    for combination in itertools.product(*source_sets):
        shops = {str(product.get("shop_id", "")) for product in combination}
        if len(shops) != 1 or "" in shops:
            continue
        rank_sum = sum(int(product.get("retrieval_rank", 9999)) for product in combination)
        ids = tuple(str(product.get("product_id", "")) for product in combination)
        key = (rank_sum, ids)
        if best is None or key < best[0]:
            best = (key, [dict(product) for product in combination])
    return best[1] if best else []


def _voucher_parameters(question: str) -> dict[str, Any] | None:
    text = str(question or "")
    budget_match = re.search(r"budget\s+is\s+only\s+`?([\d.]+)`?", text, re.I)
    threshold_match = re.search(r"total price[^\n]*exceeds\s+`?([\d.]+)`?", text, re.I)
    fixed_match = re.search(r"fixed discount of\s+`?([\d.]+)`?", text, re.I)
    percentage_match = re.search(r"percentage discount of\s+`?([\d.]+)%", text, re.I)
    cap_match = re.search(r"cap of\s+`?([\d.]+)`?", text, re.I)
    if not budget_match or not threshold_match or not (fixed_match or percentage_match):
        return None
    return {
        "budget": float(budget_match.group(1)),
        "threshold": float(threshold_match.group(1)),
        "fixed": float(fixed_match.group(1)) if fixed_match else None,
        "percentage": float(percentage_match.group(1)) / 100 if percentage_match else None,
        "cap": float(cap_match.group(1)) if cap_match else None,
        "same_shop": "only applies to the products from the same shop" in text.lower(),
    }


def _apply_voucher(
    source_sets: Sequence[Sequence[Mapping[str, Any]]],
    question: str,
) -> list[dict[str, Any]]:
    params = _voucher_parameters(question)
    if params is None or not source_sets or any(not source for source in source_sets):
        return []
    best: tuple[tuple[int, float, tuple[str, ...]], list[dict[str, Any]]] | None = None
    for combination in itertools.product(*source_sets):
        if params["same_shop"]:
            shops = {str(product.get("shop_id", "")) for product in combination}
            if len(shops) != 1 or "" in shops:
                continue
        prices = [product.get("price") for product in combination]
        if any(price is None for price in prices):
            continue
        total = sum(float(price) for price in prices)
        if total <= params["threshold"]:
            continue
        if params["fixed"] is not None:
            discount = params["fixed"]
        else:
            discount = total * params["percentage"]
            if params["cap"] is not None:
                discount = min(discount, params["cap"])
        discounted_total = total - discount
        if discounted_total > params["budget"]:
            continue
        rank_sum = sum(int(product.get("retrieval_rank", 9999)) for product in combination)
        ids = tuple(str(product.get("product_id", "")) for product in combination)
        key = (rank_sum, discounted_total, ids)
        if best is None or key < best[0]:
            best = (key, [dict(product) for product in combination])
    return best[1] if best else []


def execute_artifact_reason(
    node: Mapping[str, Any],
    source_sets: Sequence[Sequence[Mapping[str, Any]]],
    question: str,
) -> tuple[list[dict[str, Any]], str]:
    """Execute ShoppingBench Reason without allowing free-text facts into the data plane."""
    op = re.sub(r"[^a-z0-9]+", "_", str(node.get("op", "")).lower()).strip("_")
    if not source_sets or any(not source for source in source_sets):
        return [], "missing_product_set"
    if op in {"filter", "filter_constraints"}:
        return _deduplicate_products(source_sets[0]), "deterministic_filter"
    if op in {"intersect", "intersect_product"}:
        common = set(str(item.get("product_id", "")) for item in source_sets[0])
        for source in source_sets[1:]:
            common &= {str(item.get("product_id", "")) for item in source}
        return [dict(item) for item in source_sets[0] if str(item.get("product_id", "")) in common], "deterministic_intersection"
    if op in {"merge", "join_shop"}:
        return _best_same_shop_group(source_sets), "deterministic_shop_join"
    if op in {"arithmetic", "apply_voucher"}:
        return _apply_voucher(source_sets, question), "deterministic_voucher"
    if op in {"argmin", "argmax"}:
        candidates = _deduplicate_products([item for source in source_sets for item in source])
        priced = [item for item in candidates if item.get("price") is not None]
        if not priced:
            return [], "missing_price"
        selected = (min if op == "argmin" else max)(priced, key=lambda item: float(item["price"]))
        return [selected], f"deterministic_{op}"
    return [], "unsupported_artifact_op"


def serialize_product_sink(
    products: Sequence[Mapping[str, Any]],
    *,
    sink_node: Mapping[str, Any] | None = None,
) -> str:
    products = list(products)
    if str((sink_node or {}).get("type", "")) == "search":
        model_selected = [product for product in products if product.get("model_selected")]
        products = (model_selected or products)[:1]
    product_ids = [
        str(product.get("product_id", ""))
        for product in _deduplicate_products(products)
        if str(product.get("product_id", "")).isdigit()
    ]
    return f"<product_ids>{','.join(product_ids)}</product_ids>"


QUERY_REWRITE_SYSTEM_MSG = """Rewrite a dependent ShoppingBench search as one standalone query.

Return either plain search text or one compact JSON object using only q, page,
shop_id, price, sort, service, and product_ids. Replace #ID# with supplied product
facts. Preserve concrete constraints. Do not answer or add explanation.
"""

SEARCH_SYSTEM_MSG = """Extract product evidence from the supplied ShoppingBench catalog context.

When useful candidates are present, output one conclusion XML element whose text
contains the relevant catalog facts. The text must include each relevant numeric
Product ID and Shop ID exactly as shown. When another search is required, output
one rewrite XML element containing the next concrete plain-text or JSON query.

Rules:
- Use only the supplied catalog context.
- Preserve every candidate's exact Product ID and Shop ID.
- Preserve relevant price, service, SKU, and attribute values verbatim.
- State which requested constraints are satisfied, contradicted, or unverified.
- Include at most five candidates. Use one short plain-text line per candidate.
- Do not create product tags or any nested XML inside the conclusion element.
- Do not repeat a candidate or copy full descriptions.
- Never invent a product, ID, shop, price, service, SKU, or attribute.
- Never output an empty element, an instruction label, or a generic description.
- Do not add text outside the XML element.
"""

ANSWER_GENERATION_SYSTEM_MSG = SEARCH_SYSTEM_MSG

REASON_SYSTEM_MSG = """Resolve the ShoppingBench selection using only prior node conclusions.

Keep exact Product IDs and Shop IDs. Check all requested constraints. For shop
tasks require one shared Shop ID; for voucher tasks compute the discounted total
from retrieved prices and the stated voucher. Output one <conclusion> element and
do not introduce catalog facts absent from the inputs.
"""

FINAL_SYSTEM_MSG = """Return the selected ShoppingBench product IDs.

Use only products supported by resolved catalog evidence. Preserve the requested
product order. Begin with <answer><product_ids>, write only the selected numeric
IDs separated by commas, and end with </product_ids></answer>. For one product,
write one ID. Never use an example or invented ID. If no fully supported selection
exists, leave the product_ids element empty. Do not add prose.
"""

FINAL_RETRY_USER_MSG = (
    "Regenerate the answer XML. Its only child must be product_ids, containing "
    "only comma-separated numeric IDs copied from the resolved catalog evidence."
)

_PLACEHOLDER_RE = re.compile(
    r"SEARCH_QUERY_OR_JSON_FILTER|EVIDENCE_BASED_OPERATION|CATALOG_EVIDENCE|"
    r"CONCRETE_QUERY_OR_JSON|COMMA_SEPARATED_NUMERIC_IDS",
    re.IGNORECASE,
)
_PRODUCT_IDS_RE = re.compile(r"^\d+(?:,\d+)*$")


def is_placeholder_text(text: str) -> bool:
    return bool(_PLACEHOLDER_RE.search(text or ""))


def normalize_final_answer(answer: str) -> str:
    return (answer or "").strip()


def is_retrieval_failure_context(context: str) -> bool:
    normalized = (context or "").lstrip()
    return normalized.startswith(
        ("[Search API Error]", "[No Documents Found]", "[No query provided]")
    )


def validate_final_answer(answer: str) -> str | None:
    """Return a format issue, or None for valid product ID XML."""
    answer = normalize_final_answer(answer)
    if is_placeholder_text(answer):
        return "answer contains an instruction placeholder"
    try:
        root = ET.fromstring(f"<answer>{answer}</answer>")
    except ET.ParseError as exc:
        return f"malformed XML: {exc}"
    children = list(root)
    if len(children) != 1 or children[0].tag != "product_ids":
        return "expected one product_ids element"
    if list(children[0]):
        return "product_ids must contain text only"
    product_ids = (children[0].text or "").strip().replace(" ", "")
    if product_ids and not _PRODUCT_IDS_RE.fullmatch(product_ids):
        return "product_ids must be comma-separated numeric IDs"
    return None


def extract_product_ids(answer: str) -> list[str]:
    if validate_final_answer(answer) is not None:
        return []
    root = ET.fromstring(f"<answer>{normalize_final_answer(answer)}</answer>")
    value = (root.findtext("product_ids") or "").strip().replace(" ", "")
    return value.split(",") if value else []
