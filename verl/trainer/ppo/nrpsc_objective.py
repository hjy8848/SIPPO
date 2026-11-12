"""Opt-in, frozen-batch objectives for auditing Planner preference variants.

This module does not change the training launcher or rollout contract.
Masks are supplied by a separately audited node aligner.
"""
from dataclasses import dataclass, replace
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class PlanNode:
    kind: str
    node_id: str
    text: str
    references: tuple[str, ...]
    start: int
    end: int


_NODE = re.compile(r'<(search|reason)\s+id="(\d+)"\s*>(.*?)</\1>', re.I | re.S)
_REFERENCE = re.compile(r'#(\d+)#')
_WORD = re.compile(r'[a-z0-9]+')


def optional_psc_metadata(metadata) -> list[str]:
    """Retain optional rollout metadata when constructing actor minibatches."""
    keys = ("psc_chosen_eligible", "contract_valid", "runtime_lineage_valid",
            "answer_sink_supported", "outcome_reward", "workflow_valid",
            "nrpsc_tiebreak_usable", "nrpsc_tiebreak_hit")
    return [key for key in keys if key in metadata]


def validate_sibling_groups(question_ids, trajectory_ids, sample_indices, valid_rows,
                            expected_siblings: int) -> int:
    """Validate complete, distinct workflow indices before question-wise scatter."""
    if expected_siblings < 2:
        raise ValueError("NR-PSC expected sibling count must be at least two")
    if not (len(question_ids) == len(trajectory_ids) == len(sample_indices) == len(valid_rows)):
        raise ValueError("NR-PSC sibling metadata length mismatch")
    questions, trajectories = {}, {}
    for question, trajectory, sample, valid in zip(question_ids, trajectory_ids, sample_indices, valid_rows, strict=True):
        if not bool(valid):
            continue
        sample = int(sample)
        identity = (question, sample)
        previous = trajectories.setdefault(trajectory, identity)
        if previous != identity:
            raise ValueError("NR-PSC trajectory has inconsistent sibling identity")
        group = questions.setdefault(question, {})
        previous_trajectory = group.setdefault(sample, trajectory)
        if previous_trajectory != trajectory:
            raise ValueError("NR-PSC sibling index maps to multiple trajectories")
    expected = set(range(expected_siblings))
    if not questions or any(set(group) != expected for group in questions.values()):
        raise ValueError("NR-PSC requires complete sibling indices for every question")
    return len(questions)


def _tfidf_words(text):
    return _WORD.findall(_REFERENCE.sub(' REF ', text.lower()))


@dataclass(frozen=True)
class FrozenTfidf:
    idf: dict[str, float]
    unknown_idf: float | None = None

    def __call__(self, left, right):
        a, b = Counter(_tfidf_words(left)), Counter(_tfidf_words(right))
        if self.unknown_idf is None and (set(a) | set(b)) - set(self.idf):
            raise ValueError("TF-IDF scorer encountered an out-of-corpus term")
        def weight(key):
            return self.idf[key] if key in self.idf else self.unknown_idf
        denom = math.sqrt(sum((count * weight(key)) ** 2 for key, count in a.items()) *
                          sum((count * weight(key)) ** 2 for key, count in b.items()))
        if not denom:
            return 0.0
        return min(1.0, sum(count * b.get(key, 0) * weight(key) ** 2 for key, count in a.items()) / denom)


@dataclass(frozen=True)
class FrozenReliabilityTable:
    """Precalibrated weights keyed by ordered winner/loser terminal counts."""
    weights: dict[str, float]

    def __call__(self, successes: int, trials: int, other_successes: int, other_trials: int) -> float:
        key = f"{successes}/{trials}|{other_successes}/{other_trials}"
        if key not in self.weights:
            raise ValueError(f"missing pre-calibrated NR-PSC reliability weight: {key}")
        weight = float(self.weights[key])
        if not math.isfinite(weight) or not 0 <= weight <= 1:
            raise ValueError(f"invalid NR-PSC reliability weight: {key}")
        return weight


def frozen_tfidf(documents: list[str]) -> FrozenTfidf:
    """Fit deterministic unigram TF-IDF once before distributed scattering."""
    if not documents:
        raise ValueError("TF-IDF requires at least one document")
    df = Counter(term for text in documents for term in set(_tfidf_words(text)))
    return FrozenTfidf({term: math.log((1 + len(documents)) / (1 + count)) + 1
                        for term, count in df.items()}, math.log(1 + len(documents)) + 1)


def load_frozen_tfidf(path: str) -> tuple[FrozenTfidf, str]:
    """Load a frozen, provenance-bearing IDF artifact and its file SHA-256."""
    data = Path(path).read_bytes()
    artifact = json.loads(data)
    if artifact.get("schema") != "nrpsc_tfidf_v1":
        raise ValueError("unsupported NR-PSC IDF artifact schema")
    count = artifact.get("document_count")
    values = artifact.get("idf")
    if not isinstance(count, int) or isinstance(count, bool) or count <= 0 or not isinstance(values, dict):
        raise ValueError("invalid NR-PSC IDF document count or vocabulary")
    unknown = float(artifact.get("unknown_idf", float('nan')))
    if not math.isclose(unknown, math.log(1 + count) + 1, rel_tol=1e-12):
        raise ValueError("invalid NR-PSC unknown-token smoothing weight")
    if not isinstance(artifact.get("corpus_sha256"), str) or not re.fullmatch(r'[0-9a-f]{64}', artifact['corpus_sha256']):
        raise ValueError("NR-PSC IDF artifact needs a corpus SHA-256")
    if not values or any(not isinstance(key, str) or not key or isinstance(value, bool)
                         or not isinstance(value, (float, int)) or not math.isfinite(value)
                         or value < 1 or value > unknown for key, value in values.items()):
        raise ValueError("invalid NR-PSC IDF weights")
    return FrozenTfidf({key: float(value) for key, value in values.items()}, unknown), hashlib.sha256(data).hexdigest()


def parse_plan(text: str) -> tuple[PlanNode, ...]:
    """Parse supported nodes and reject malformed, forward or dangling edges."""
    matches = list(_NODE.finditer(text))
    if not matches or len(re.findall(r'<(?:search|reason)\b', text, re.I)) != len(matches):
        raise ValueError("missing or malformed plan nodes")
    seen = set()
    result = []
    for match in matches:
        node_id = match.group(2)
        refs = tuple(_REFERENCE.findall(match.group(3)))
        if node_id in seen or any(ref not in seen for ref in refs):
            raise ValueError("duplicate or unresolved node reference")
        result.append(PlanNode(match.group(1).lower(), node_id, match.group(3), refs,
                               match.start(), match.end()))
        seen.add(node_id)
    return tuple(result)


MAX_ALIGNMENT_NODES = 8  # exact matching is exponential; plans above this size are excluded, never allowed to abort a batch


def align_nodes(left: tuple[PlanNode, ...], right: tuple[PlanNode, ...],
                similarity, threshold: float) -> tuple[tuple[int, int], ...]:
    """Maximum one-to-one DAG-compatible matching under a frozen scorer.

    Pairwise matching is deliberately not treated as a transitive class relation.
    """
    if not 0 <= threshold <= 1 or len(left) > MAX_ALIGNMENT_NODES or len(right) > MAX_ALIGNMENT_NODES:
        raise ValueError("invalid threshold or plan exceeds bounded alignment size")
    best = (0, 0.0, ())

    def search(index, matched, used, score):
        nonlocal best
        if index == len(left):
            mapping = {left[i].node_id: right[j].node_id for i, j in matched}
            for i, j in matched:
                source, target = left[i].references, right[j].references
                if any(ref not in mapping for ref in source):
                    return
                if sorted(mapping[ref] for ref in source) != sorted(target):
                    return
            candidate = (len(matched), score, tuple(matched))
            if candidate[:2] > best[:2]:
                best = candidate
            return
        search(index + 1, matched, used, score)
        for j, node in enumerate(right):
            if j in used or left[index].kind != node.kind:
                continue
            value = float(similarity(left[index].text, node.text))
            if not 0 <= value <= 1:
                raise ValueError("similarity must be finite and within [0, 1]")
            if value >= threshold:
                search(index + 1, matched + [(index, j)], used | {j}, score + value)

    search(0, [], set(), 0.0)
    return best[2]


@dataclass(frozen=True)
class PreferencePair:
    question: object
    winner: int
    loser: int
    winner_nodes: tuple[torch.Tensor, ...]
    loser_nodes: tuple[torch.Tensor, ...]
    weight: float = 1.0
    class_pair: object = None


@dataclass(frozen=True)
class PlanView:
    nodes: tuple[PlanNode, ...]
    node_masks: tuple[torch.Tensor, ...]
    prompt_ids: tuple[int, ...]
    response_ids: tuple[int, ...]
    end_mask: torch.Tensor


@dataclass(frozen=True)
class NodeComparison:
    kind: str
    left_nodes: tuple[torch.Tensor, ...]
    right_nodes: tuple[torch.Tensor, ...]


@dataclass(frozen=True)
class PlanObservation:
    question: object
    row: int
    view: PlanView
    successes: int
    trials: int
    eligible: bool = True
    # Optional secondary outcome (e.g. "did any retrieved document contain the gold answer?"), used only to break
    # a tie in the primary (EM) class comparison in `construct_node_pairs`. None means "not available for this
    # row" (binary-answer questions, or questions where the answer string cannot be checked against retrieval);
    # such rows never contribute to the tie-break, the same as if tie-break were disabled for that comparison.
    secondary_successes: int | None = None
    secondary_trials: int | None = None


def cluster_plan_observations(members: list[PlanObservation], similarity, threshold: float):
    """Deterministic complete-link plan classes; no transitivity assumption."""
    if len({member.view.prompt_ids for member in members}) > 1:
        raise ValueError("NR-PSC requires byte-identical Planner prompt tokens within a question")
    classes = []
    for member in sorted(members, key=lambda item: item.row):
        for group in classes:
            if all(compare_plans(member.view, other.view, similarity, threshold).kind == "same"
                   for other in group):
                group.append(member)
                break
        else:
            classes.append([member])
    return classes


def select_replay_rows(observations: list[PlanObservation], similarity, threshold: float,
                       budget: int) -> list[int]:
    """Allocate one additional downstream run to classes with <2 observations.

    Mixed-EM groups have priority; uniform groups require >=2 distinct classes.
    The deterministic round-robin avoids consuming a whole budget on one group.
    """
    if budget < 0:
        raise ValueError("negative replay budget")
    questions = {}
    for member in observations:
        if member.eligible:
            questions.setdefault(member.question, []).append(member)
    buckets = [[], []]
    for question in sorted(questions, key=str):
        members = questions[question]
        classes = cluster_plan_observations(members, similarity, threshold)
        if len(classes) < 2:
            continue
        outcomes = {member.successes / member.trials for member in members}
        priority = 0 if len(outcomes) > 1 else 1
        candidates = [group[0].row for group in classes if sum(x.trials for x in group) < 2]
        if candidates:
            buckets[priority].append(candidates)
    selected = []
    for bucket in buckets:
        while len(selected) < budget and any(bucket):
            for candidates in bucket:
                if candidates and len(selected) < budget:
                    selected.append(candidates.pop(0))
    return selected


def construct_node_pairs(observations: list[PlanObservation], similarity, threshold: float,
                         reliability=None, *, tiebreak_weight: float = 0.0) -> tuple[list[PreferencePair], dict]:
    """Group fully aligned plans by complete linkage and build class-level pairs.

    `successes`/`trials` must describe independently executed terminal-EM
    outcomes of that specific plan. `reliability`, if provided, maps
    (winner_successes, winner_trials, loser_successes, loser_trials) to a
    pre-calibrated bounded weight. No model or reward is fit here.

    `tiebreak_weight` (default 0.0, i.e. off: identical behaviour and counts to the original function) scales a
    class pair whose EM success rates tie exactly but whose ``secondary_successes``/``secondary_trials`` differ.
    This never overrides an EM difference, never fires on a class with no secondary data, and is a separate,
    independently tunable weight from the main NR-PSC loss weight -- it adds coverage on EM-tied comparisons
    (most visibly the all-fail questions where EM gives no signal at all) without rescaling the EM-based pairs.
    """
    if tiebreak_weight < 0 or not math.isfinite(tiebreak_weight):
        raise ValueError("invalid tiebreak_weight")
    questions = {}
    for obs in observations:
        if obs.trials < 1 or not 0 <= obs.successes <= obs.trials:
            raise ValueError("invalid terminal EM execution counts")
        if obs.secondary_trials is not None and (
            obs.secondary_trials < 1 or obs.secondary_successes is None
            or not 0 <= obs.secondary_successes <= obs.secondary_trials
        ):
            raise ValueError("invalid secondary execution counts")
        if obs.eligible:
            questions.setdefault(obs.question, []).append(obs)
    result = []
    counts = {"questions": len(questions), "classes": 0, "same_class_members": 0,
              "class_pairs": 0, "informative_class_pairs": 0, "em_informative_class_pairs": 0,
              "tiebreak_informative_class_pairs": 0, "G": 0, "S": 0, "N": 0,
              "candidate_counts": {}}
    for question, members in questions.items():
        classes = cluster_plan_observations(members, similarity, threshold)
        counts["same_class_members"] += len(members) - len(classes)
        counts["classes"] += len(classes)
        counts["candidate_counts"][question] = len(classes) * (len(classes) - 1) // 2
        for i, left in enumerate(classes):
            for j in range(i + 1, len(classes)):
                right = classes[j]
                counts["class_pairs"] += 1
                ls, ln = sum(x.successes for x in left), sum(x.trials for x in left)
                rs, rn = sum(x.successes for x in right), sum(x.trials for x in right)
                via_tiebreak = False
                if ls * rn == rs * ln:
                    if tiebreak_weight <= 0:
                        continue
                    sec_left = [x for x in left if x.secondary_trials is not None]
                    sec_right = [x for x in right if x.secondary_trials is not None]
                    if not sec_left or not sec_right:
                        continue
                    ls, ln = sum(x.secondary_successes for x in sec_left), sum(x.secondary_trials for x in sec_left)
                    rs, rn = sum(x.secondary_successes for x in sec_right), sum(x.secondary_trials for x in sec_right)
                    if ls * rn == rs * ln:
                        continue
                    via_tiebreak = True
                winner, loser, ws, wn, ls, ln = ((left, right, ls, ln, rs, rn)
                                                if ls * rn > rs * ln
                                                else (right, left, rs, rn, ls, ln))
                weight = 1.0 if reliability is None else float(reliability(ws, wn, ls, ln))
                if not 0 <= weight <= 1 or not math.isfinite(weight):
                    raise ValueError("invalid reliability weight")
                if via_tiebreak:
                    weight *= tiebreak_weight
                if weight == 0:
                    continue
                counts["informative_class_pairs"] += 1
                counts["tiebreak_informative_class_pairs" if via_tiebreak else "em_informative_class_pairs"] += 1
                for win in winner:
                    for lose in loser:
                        comparison = compare_plans(win.view, lose.view, similarity, threshold)
                        if comparison.kind == "same":
                            # Complete-link classes need not make all cross-class
                            # pairs distinct when similarity is non-transitive.
                            continue
                        counts[comparison.kind] += 1
                        result.append(PreferencePair(question, win.row, lose.row,
                                                     comparison.left_nodes, comparison.right_nodes,
                                                     weight, (i, j)))
    return result, counts


def compare_plans(left: PlanView, right: PlanView, similarity, threshold: float) -> NodeComparison:
    """Build G/S/N decision masks, requiring actual token-prefix equality for S."""
    matches = align_nodes(left.nodes, right.nodes, similarity, threshold)
    used_left, used_right = {i for i, _ in matches}, {j for _, j in matches}
    left_unmatched = tuple(left.node_masks[i] for i in range(len(left.nodes)) if i not in used_left)
    right_unmatched = tuple(right.node_masks[i] for i in range(len(right.nodes)) if i not in used_right)
    if not left_unmatched and not right_unmatched:
        return NodeComparison("same", (), ())
    if left_unmatched and right_unmatched:
        return NodeComparison("G", left_unmatched, right_unmatched)
    longer, shorter = (left, right) if left_unmatched else (right, left)
    used_long = used_left if left_unmatched else used_right
    # The shorter plan's nodes must appear as an ordered prefix of the longer.
    ordered_pairs = tuple(sorted(matches)) if left_unmatched else tuple(sorted((j, i) for i, j in matches))
    ordered = (len(shorter.nodes) == len(used_long) and
               ordered_pairs == tuple((i, i) for i in range(len(shorter.nodes))))
    end = shorter.end_mask.bool()
    if end.ndim != 1 or len(end) != len(shorter.response_ids):
        raise ValueError("end mask does not match response")
    end_indices = torch.nonzero(end).flatten()
    split = int(end_indices[0]) if len(end_indices) else -1
    aligned = (ordered and split >= 0 and longer.prompt_ids == shorter.prompt_ids and
               longer.response_ids[:split] == shorter.response_ids[:split])
    if aligned:
        return (NodeComparison("S", left_unmatched, (end,)) if left_unmatched
                else NodeComparison("S", (end,), right_unmatched))
    return NodeComparison("N", left_unmatched, right_unmatched)


def node_token_mask(offsets: list[tuple[int, int]], start: int, end: int) -> torch.Tensor:
    """Assign a token to the node containing its starting character."""
    if start < 0 or end <= start:
        raise ValueError("invalid node span")
    result = []
    for left, right in offsets:
        result.append(left < right and start <= left < end)
    if not any(result):
        raise ValueError("node span has no aligned tokens")
    return torch.tensor(result, dtype=torch.bool)


def planner_node_masks(tokenizer, response_ids: list[int]) -> tuple[tuple[PlanNode, ...], tuple[torch.Tensor, ...]]:
    """Decode and align node spans only when token IDs round-trip exactly.

    IDs and returned masks represent unpadded response tokens. A tokenizer
    whose decoded text cannot be re-encoded exactly must not be used to infer
    character-to-token correspondence.
    """
    text = tokenizer.decode(response_ids, skip_special_tokens=False,
                            clean_up_tokenization_spaces=False)
    encoding = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    ids = list(encoding["input_ids"])
    if ids != list(response_ids):
        raise ValueError("response token IDs do not round-trip through tokenizer")
    offsets = [(int(left), int(right)) for left, right in encoding["offset_mapping"]]
    if len(offsets) != len(response_ids):
        raise ValueError("tokenizer returned inconsistent offset count")
    nodes = parse_plan(text)
    return nodes, tuple(node_token_mask(offsets, node.start, node.end) for node in nodes)


def planner_plan_view(tokenizer, prompt_ids: list[int], response_ids: list[int]) -> PlanView:
    """Construct one aligned Planner view, including closing-plan/EOS tokens."""
    text = tokenizer.decode(response_ids, skip_special_tokens=False,
                            clean_up_tokenization_spaces=False)
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    if list(encoded["input_ids"]) != list(response_ids):
        raise ValueError("response token IDs do not round-trip through tokenizer")
    offsets = [(int(a), int(b)) for a, b in encoded["offset_mapping"]]
    if len(offsets) != len(response_ids):
        raise ValueError("tokenizer returned inconsistent offset count")
    nodes = parse_plan(text)
    masks = tuple(node_token_mask(offsets, node.start, node.end) for node in nodes)
    endings = list(re.finditer(r'</plan\s*>', text, re.I))
    if len(endings) != 1 or endings[0].start() < nodes[-1].end:
        raise ValueError("plan must have one closing marker after its nodes")
    marker = endings[0]
    end_mask = node_token_mask(offsets, marker.start(), marker.end())
    eos = getattr(tokenizer, 'eos_token_id', None)
    if eos is not None:
        for index, ((start, _), token) in enumerate(zip(offsets, response_ids, strict=True)):
            if token == eos and start >= marker.end():
                end_mask[index] = True
    return PlanView(nodes, masks, tuple(prompt_ids), tuple(response_ids), end_mask)


def prepare_plan_views(tokenizer, responses: torch.Tensor, response_mask: torch.Tensor,
                       prompts: torch.Tensor, attention_mask: torch.Tensor,
                       roles) -> tuple[np.ndarray, dict[str, int]]:
    """Create row-aligned, serializable Planner views before actor scattering."""
    if responses.shape != response_mask.shape or len(roles) != len(responses):
        raise ValueError("inconsistent Planner batch shapes")
    if len(prompts) != len(responses) or attention_mask.shape[0] != len(responses):
        raise ValueError("inconsistent prompt batch shapes")
    rows = np.empty(len(responses), dtype=object)
    rows[:] = None
    counts = Counter()
    prompt_width = prompts.shape[1]
    for row, role in enumerate(roles):
        if str(role) != 'PlanningAgent':
            continue
        counts['planner_rows'] += 1
        ids = responses[row][response_mask[row].bool()].tolist()
        prompt = prompts[row][attention_mask[row][:prompt_width].bool()].tolist()
        try:
            view = planner_plan_view(tokenizer, prompt, ids)
        except ValueError:
            counts['parse_or_alignment_failures'] += 1
        else:
            if len(view.nodes) > MAX_ALIGNMENT_NODES:
                counts['oversize_plans'] += 1  # excluded from NR-PSC pairing, reported in aux/nrpsc/oversize_plans
                continue
            rows[row] = view
            counts['usable_rows'] += 1
    return rows, dict(counts)


def global_question_scale(local_questions: int, global_questions: int, world_size: int) -> float:
    """Scale a local all-question mean before distributed gradient averaging."""
    if local_questions < 0 or global_questions < local_questions or world_size < 1:
        raise ValueError("invalid distributed question counts")
    return local_questions * world_size / global_questions if global_questions else 0.0


def reward_eligibility_mask(em: torch.Tensor, reward: torch.Tensor,
                            valid: torch.Tensor) -> torch.Tensor:
    """Filter positives only; rejected trajectories remain valid comparisons.

    The caller must supply the exact binary terminal reward used by W, not a
    shaped reward. Missing metadata must be rejected by the caller.
    """
    if em.shape != reward.shape or em.shape != valid.shape:
        raise ValueError("reward, EM and validity must have the same shape")
    if not torch.isfinite(em).all() or not torch.isfinite(reward).all():
        raise ValueError("non-finite reward or EM")
    if not (((em == 0) | (em == 1)).all() and ((reward == 0) | (reward == 1)).all()):
        raise ValueError("eligibility expects binary terminal EM and W reward")
    return (em == 0) | ((reward == 1) & valid.bool())


def actual_reward_eligibility(metadata):
    """Intersect existing PSC eligibility with the actual W terminal reward."""
    required = ("outcome_exact", "outcome_reward", "workflow_valid")
    missing = [key for key in required if key not in metadata]
    if missing:
        raise ValueError(f"PSC reward eligibility is missing metadata: {missing}")
    em = torch.as_tensor(metadata["outcome_exact"], dtype=torch.float64)
    reward = torch.as_tensor(metadata["outcome_reward"], dtype=torch.float64)
    valid = torch.as_tensor(metadata["workflow_valid"], dtype=torch.bool)
    mask = reward_eligibility_mask(em, reward, valid)
    if "psc_chosen_eligible" in metadata:
        existing = torch.as_tensor(metadata["psc_chosen_eligible"], dtype=torch.bool)
        if existing.shape != mask.shape:
            raise ValueError("PSC eligibility has inconsistent shape")
        mask &= existing
    return mask.numpy()


def _decision_score(log_probs: torch.Tensor, nodes: tuple[torch.Tensor, ...]) -> torch.Tensor:
    if not nodes:
        raise ValueError("both sides must contain at least one decision")
    values = []
    for node in nodes:
        mask = node.to(device=log_probs.device, dtype=torch.bool)
        if mask.shape != log_probs.shape or not mask.any():
            raise ValueError("node masks must match the sequence and be nonempty")
        values.append(log_probs[mask].mean())
    return torch.stack(values).mean()


def node_preference_loss(log_probs: torch.Tensor, pairs: list[PreferencePair],
                         *, question_count: int, beta: float = 1.0,
                         candidate_counts: dict | None = None) -> torch.Tensor:
    """Mean pair loss per question, normalized by *all* questions in batch.

    A pair consists of two independently masked decision sets. Callers must
    establish role, token boundaries, legality and observed preference first.
    """
    if log_probs.ndim != 2 or question_count < 1 or beta <= 0:
        raise ValueError("invalid log_probs, question_count or beta")
    grouped = {}
    for pair in pairs:
        if not 0 <= pair.winner < len(log_probs) or not 0 <= pair.loser < len(log_probs):
            raise ValueError("pair row outside batch")
        if not 0 <= pair.weight <= 1:
            raise ValueError("preference weight must be bounded in [0, 1]")
        if pair.winner_nodes and pair.loser_nodes:
            win = _decision_score(log_probs[pair.winner], pair.winner_nodes)
            lose = _decision_score(log_probs[pair.loser], pair.loser_nodes)
            loss = F.softplus(-beta * (win - lose))
        elif pair.winner_nodes:
            win = _decision_score(log_probs[pair.winner], pair.winner_nodes)
            loss = F.softplus(-beta * (win - win.detach()))
        elif pair.loser_nodes:
            lose = _decision_score(log_probs[pair.loser], pair.loser_nodes)
            loss = F.softplus(beta * (lose - lose.detach()))
        else:
            raise ValueError("pair must contain at least one decision")
        groups = grouped.setdefault(pair.question, {})
        class_pair = pair.class_pair if pair.class_pair is not None else (pair.winner, pair.loser)
        groups.setdefault(class_pair, []).append(pair.weight * loss)
    if len(grouped) > question_count:
        raise ValueError("more informative questions than batch questions")
    if not grouped:
        return log_probs.sum() * 0
    total = log_probs.sum() * 0
    for question, class_pairs in grouped.items():
        denominator = (len(class_pairs) if candidate_counts is None
                       else int(candidate_counts.get(question, 0)))
        if denominator < len(class_pairs):
            raise ValueError("candidate class pair denominator is too small")
        total = total + sum(torch.stack(losses).mean() for losses in class_pairs.values()) / denominator
    return total / question_count


def node_preference_coefficients(log_probs: torch.Tensor, pairs: list[PreferencePair],
                                 *, question_count: int, beta: float = 1.0,
                                 candidate_counts: dict | None = None) -> torch.Tensor:
    """Exact first derivative at the current policy, before actor DP scaling."""
    with torch.enable_grad():
        values = log_probs.detach().float().requires_grad_(True)
        loss = node_preference_loss(values, pairs, question_count=question_count,
                                    beta=beta, candidate_counts=candidate_counts)
        return torch.autograd.grad(loss, values)[0].detach()


def batch_node_coefficients(log_probs: torch.Tensor, response_mask: torch.Tensor,
                            observations: list[PlanObservation], similarity, threshold: float,
                            *, question_count: int, beta: float = 1.0, reliability=None,
                            tiebreak_weight: float = 0.0):
    """Build full response-token coefficients from pre-aligned Planner observations.

    Each observation must name a unique valid actor row. Batch padding and
    duplicate-workflow semantics must be handled by the caller.
    """
    if log_probs.ndim != 2 or response_mask.shape != log_probs.shape:
        raise ValueError("response_mask and log_probs must be matching matrices")
    rows = [obs.row for obs in observations]
    if len(set(rows)) != len(rows):
        raise ValueError("duplicate planner rows require explicit normalization")
    pairs, counts = construct_node_pairs(observations, similarity, threshold, reliability,
                                         tiebreak_weight=tiebreak_weight)
    def padded(row, nodes):
        if not 0 <= row < len(log_probs):
            raise ValueError("planner row outside batch")
        active = response_mask[row].bool().cpu()
        result = []
        for node in nodes:
            node = node.bool().cpu()
            if node.ndim != 1 or len(node) != int(active.sum()):
                raise ValueError("node mask does not match unpadded response")
            full = torch.zeros_like(active)
            full[active] = node
            result.append(full)
        return tuple(result)
    padded_pairs = [replace(pair, winner_nodes=padded(pair.winner, pair.winner_nodes),
                            loser_nodes=padded(pair.loser, pair.loser_nodes)) for pair in pairs]
    coeffs = node_preference_coefficients(log_probs, padded_pairs, question_count=question_count,
                                          beta=beta, candidate_counts=counts["candidate_counts"])
    return coeffs, counts


def shared_prefix_masks(winner_ids: torch.Tensor, loser_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Remove only an exact shared token prefix; caller verifies identical prompt."""
    if winner_ids.ndim != 1 or loser_ids.ndim != 1:
        raise ValueError("expected unpadded token sequences")
    length = min(len(winner_ids), len(loser_ids))
    prefix = 0
    while prefix < length and winner_ids[prefix].item() == loser_ids[prefix].item():
        prefix += 1
    win = torch.arange(len(winner_ids), device=winner_ids.device) >= prefix
    lose = torch.arange(len(loser_ids), device=loser_ids.device) >= prefix
    return win, lose


def reward_eligible(em: float, reward: float, workflow_valid: bool) -> bool:
    """A positive is eligible only when binary EM and actual reward agree."""
    return bool(workflow_valid and em == 1.0 and reward >= 1.0)
