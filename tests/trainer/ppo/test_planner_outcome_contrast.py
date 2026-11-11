from __future__ import annotations

import importlib.util
import numpy as np
from pathlib import Path
import sys

import pytest
import torch
import torch.nn.functional as F


MODULE_PATH = (
    Path(__file__).resolve().parents[3]
    / "verl"
    / "trainer"
    / "ppo"
    / "planner_outcome_contrast.py"
)
SPEC = importlib.util.spec_from_file_location("planner_outcome_contrast", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)
build_planner_outcome_contrast_coefficients = (
    MODULE.build_planner_outcome_contrast_coefficients
)
planner_outcome_contrast_loss = MODULE.planner_outcome_contrast_loss
summarize_parameter_gradient_audit = MODULE.summarize_parameter_gradient_audit


def make_batch(outcomes=(1.0, 0.0, 1.0, 0.0), *, duplicate_first=False):
    question_ids = []
    trajectory_ids = []
    roles = []
    workflow_steps = []
    row_outcomes = []
    masks = []
    logps = []
    for index, outcome in enumerate(outcomes):
        rows = [
            ("PlanningAgent", 0, [1, 1, 1], [-0.2 - index * 0.1, -0.3, -0.4]),
            ("ReasonerAgent", 1, [1, 1, 0], [-0.5, -0.6, 0.0]),
        ]
        if duplicate_first and index == 0:
            rows.insert(1, rows[0])
        for role, step, mask, logp in rows:
            question_ids.append("q0")
            trajectory_ids.append(f"w{index}")
            roles.append(role)
            workflow_steps.append(step)
            row_outcomes.append(outcome)
            masks.append(mask)
            logps.append(logp)
    return {
        "response_mask": torch.tensor(masks, dtype=torch.float32),
        "old_log_probs": torch.tensor(logps, dtype=torch.float32),
        "question_ids": np.asarray(question_ids, dtype=object),
        "trajectory_ids": np.asarray(trajectory_ids, dtype=object),
        "roles": np.asarray(roles, dtype=object),
        "outcomes": np.asarray(row_outcomes, dtype=np.float64),
        "workflow_steps": np.asarray(workflow_steps, dtype=np.int64),
    }


def test_coefficients_match_pairwise_logistic_gradient_and_only_touch_planner():
    batch = make_batch()
    old = batch["old_log_probs"].clone().requires_grad_(True)
    result = build_planner_outcome_contrast_coefficients(
        **{**batch, "old_log_probs": old.detach()}, beta=1.3
    )
    planner_rows = np.flatnonzero(batch["roles"] == "PlanningAgent").tolist()
    scores = [
        old[row][batch["response_mask"][row].bool()].mean() for row in planner_rows
    ]
    direct = torch.stack(
        [F.softplus(-1.3 * (scores[chosen] - scores[rejected]))
         for chosen in (0, 2) for rejected in (1, 3)]
    ).mean()
    direct_grad = torch.autograd.grad(direct, old)[0]

    assert torch.allclose(result.coefficients, direct_grad, atol=1e-7, rtol=0.0)
    assert result.metrics["aux/planner_outcome_contrast/pair_count"] == 4.0
    assert result.metrics["aux/planner_outcome_contrast/non_planner_nonzero_tokens"] == 0.0
    reasoner_rows = np.flatnonzero(batch["roles"] == "ReasonerAgent").tolist()
    assert torch.count_nonzero(result.coefficients[reasoner_rows]) == 0


def test_linearized_loss_has_the_exact_current_policy_gradient():
    batch = make_batch()
    result = build_planner_outcome_contrast_coefficients(**batch)
    current = batch["old_log_probs"].clone().requires_grad_(True)
    loss = planner_outcome_contrast_loss(current, result.coefficients)
    gradient = torch.autograd.grad(loss, current)[0]
    assert torch.equal(gradient, result.coefficients)


def test_all_question_weighting_counts_uniform_groups():
    batch = make_batch(outcomes=(1.0, 0.0, 0.0, 0.0))
    batch['question_ids'][4:] = 'q1'
    informative = build_planner_outcome_contrast_coefficients(**batch)
    all_questions = build_planner_outcome_contrast_coefficients(**batch, question_weighting='all_questions')
    assert all_questions.metrics['aux/planner_outcome_contrast/raw_question_count'] == 2
    assert torch.allclose(all_questions.coefficients * 2, informative.coefficients)


def test_all_question_weighting_includes_groups_without_planner_rows():
    batch = make_batch(outcomes=(1.0, 0.0, 0.0, 0.0))
    batch['question_ids'][4:] = 'q1'
    batch['roles'][4:] = 'ReasonerAgent'
    informative = build_planner_outcome_contrast_coefficients(**batch)
    all_questions = build_planner_outcome_contrast_coefficients(**batch, question_weighting='all_questions')
    assert all_questions.metrics['aux/planner_outcome_contrast/raw_question_count'] == 1
    assert all_questions.metrics['aux/planner_outcome_contrast/all_question_count'] == 2
    assert torch.allclose(all_questions.coefficients * 2, informative.coefficients)


def test_shared_prefix_mask_uses_exact_prompt_and_token_ids():
    batch = make_batch(outcomes=(1.0, 0.0))
    responses = torch.tensor([[1, 2, 3], [1, 0, 0], [1, 2, 4], [1, 0, 0]])
    prompts = torch.tensor([[7, 8]] * 4)
    kwargs = dict(remove_shared_prefix=True, response_token_ids=responses,
                  prompt_token_ids=prompts, prompt_attention_mask=torch.ones_like(prompts))
    result = build_planner_outcome_contrast_coefficients(**batch, **kwargs)
    assert result.coefficients[0, :2].eq(0).all()
    assert result.coefficients[2, :2].eq(0).all()
    assert result.coefficients[0, 2] < 0 < result.coefficients[2, 2]
    prompts[2, 0] = 9
    with pytest.raises(ValueError, match='identical Planner prompts'):
        build_planner_outcome_contrast_coefficients(**batch, **kwargs)


def test_shared_prefix_identical_sequences_produce_no_pair():
    batch = make_batch(outcomes=(1.0, 0.0))
    responses = torch.tensor([[1, 2, 3], [1, 0, 0], [1, 2, 3], [1, 0, 0]])
    prompts = torch.tensor([[7, 8]] * 4)
    result = build_planner_outcome_contrast_coefficients(
        **batch, remove_shared_prefix=True, response_token_ids=responses,
        prompt_token_ids=prompts, prompt_attention_mask=torch.ones_like(prompts))
    assert result.metrics['aux/planner_outcome_contrast/shared_prefix_empty_pairs'] == 1
    assert result.coefficients.eq(0).all()


def test_objective_ablation_variants_change_pairing_or_length_geometry():
    batch = make_batch()
    base = build_planner_outcome_contrast_coefficients(**batch)
    no_length = build_planner_outcome_contrast_coefficients(
        **batch, length_normalization=False
    )
    pair_weighted = build_planner_outcome_contrast_coefficients(
        **batch, question_weighting="pair"
    )
    cross = build_planner_outcome_contrast_coefficients(
        **{**batch, "question_ids": np.asarray(["q0", "q0", "q0", "q0", "q1", "q1", "q1", "q1"], dtype=object)},
        pairing_mode="cross_question",
        question_weighting="pair",
    )

    assert base.metrics["aux/planner_outcome_contrast/pair_count"] == 4.0
    assert no_length.metrics["aux/planner_outcome_contrast/pair_count"] == 4.0
    assert pair_weighted.metrics["aux/planner_outcome_contrast/pair_count"] == 4.0
    assert torch.isfinite(no_length.coefficients).all()
    assert torch.isfinite(pair_weighted.coefficients).all()
    assert cross.metrics["aux/planner_outcome_contrast/pair_count"] >= 1.0


def test_one_random_pair_per_question_is_deterministic():
    batch = make_batch()
    batch["question_ids"] = np.asarray(["q0", "q0", "q0", "q0", "q1", "q1", "q1", "q1"], dtype=object)
    first = build_planner_outcome_contrast_coefficients(
        **batch, pair_selection="one_random", shuffle_seed=23
    )
    second = build_planner_outcome_contrast_coefficients(
        **batch, pair_selection="one_random", shuffle_seed=23
    )
    assert first.metrics["aux/planner_outcome_contrast/pair_count"] == 2.0
    torch.testing.assert_close(first.coefficients, second.coefficients)


@pytest.mark.parametrize("outcomes", [(0.0, 0.0, 0.0, 0.0), (1.0, 1.0, 1.0, 1.0)])
def test_constant_outcome_questions_have_no_signal(outcomes):
    result = build_planner_outcome_contrast_coefficients(**make_batch(outcomes))
    assert torch.count_nonzero(result.coefficients) == 0
    assert result.metrics["aux/planner_outcome_contrast/pair_count"] == 0.0
    assert result.metrics["aux/planner_outcome_contrast/informative_question_count"] == 0.0


def test_ineligible_em_positive_is_excluded_instead_of_becoming_rejected():
    batch = make_batch()
    eligibility = np.ones(len(batch["outcomes"]), dtype=np.bool_)
    eligibility[batch["trajectory_ids"] == "w0"] = False
    result = build_planner_outcome_contrast_coefficients(
        **batch,
        chosen_eligibility=eligibility,
    )
    w0_rows = np.flatnonzero(batch["trajectory_ids"] == "w0").tolist()
    assert torch.count_nonzero(result.coefficients[w0_rows]) == 0
    assert result.metrics["aux/planner_outcome_contrast/pair_count"] == 2.0
    assert result.metrics[
        "aux/planner_outcome_contrast/ineligible_positive_workflow_count"
    ] == 1.0


def test_coverage_metrics_separate_raw_mixed_em_from_cp_eligibility():
    batch = make_batch(outcomes=(1.0, 0.0, 0.0, 0.0))
    positive_rows = batch["trajectory_ids"] == "w0"
    eligibility = np.ones(len(batch["outcomes"]), dtype=np.bool_)
    eligibility[positive_rows] = False
    contract_validity = np.ones(len(batch["outcomes"]), dtype=np.bool_)
    contract_validity[positive_rows] = False
    lineage_validity = np.ones(len(batch["outcomes"]), dtype=np.bool_)
    sink_support = np.ones(len(batch["outcomes"]), dtype=np.bool_)

    result = build_planner_outcome_contrast_coefficients(
        **batch,
        chosen_eligibility=eligibility,
        contract_validity=contract_validity,
        runtime_lineage_validity=lineage_validity,
        answer_sink_support=sink_support,
    )

    prefix = "aux/planner_outcome_contrast/"
    assert result.metrics[prefix + "raw_planner_workflow_count"] == 4.0
    assert result.metrics[prefix + "raw_positive_workflow_count"] == 1.0
    assert result.metrics[prefix + "raw_eligible_workflow_count"] == 3.0
    assert result.metrics[prefix + "raw_eligible_positive_workflow_count"] == 0.0
    assert result.metrics[prefix + "raw_mixed_question_count"] == 1.0
    assert result.metrics[prefix + "contract_valid_workflow_count"] == 3.0
    assert result.metrics[prefix + "positive_contract_valid_workflow_count"] == 0.0
    assert result.metrics[prefix + "runtime_lineage_valid_workflow_count"] == 4.0
    assert result.metrics[prefix + "positive_runtime_lineage_valid_workflow_count"] == 1.0
    assert result.metrics[prefix + "answer_sink_supported_workflow_count"] == 4.0
    assert result.metrics[prefix + "positive_answer_sink_supported_workflow_count"] == 1.0
    assert result.metrics[prefix + "eligibility_component_mismatch_workflow_count"] == 0.0
    assert result.metrics[prefix + "ineligible_positive_workflow_count"] == 1.0
    assert result.metrics[prefix + "pair_count"] == 0.0


def test_coverage_metrics_detect_eligibility_adapter_mismatch():
    batch = make_batch(outcomes=(1.0, 0.0, 0.0, 0.0))
    eligibility = np.ones(len(batch["outcomes"]), dtype=np.bool_)
    contract_validity = np.ones(len(batch["outcomes"]), dtype=np.bool_)
    lineage_validity = np.ones(len(batch["outcomes"]), dtype=np.bool_)
    sink_support = np.ones(len(batch["outcomes"]), dtype=np.bool_)
    positive_rows = batch["trajectory_ids"] == "w0"
    eligibility[positive_rows] = False

    result = build_planner_outcome_contrast_coefficients(
        **batch,
        chosen_eligibility=eligibility,
        contract_validity=contract_validity,
        runtime_lineage_validity=lineage_validity,
        answer_sink_support=sink_support,
    )

    assert result.metrics[
        "aux/planner_outcome_contrast/eligibility_component_mismatch_workflow_count"
    ] == 1.0


def test_dp_padding_duplicates_do_not_change_the_logical_gradient():
    base = build_planner_outcome_contrast_coefficients(**make_batch())
    duplicated_batch = make_batch(duplicate_first=True)
    duplicated = build_planner_outcome_contrast_coefficients(**duplicated_batch)
    planner_rows = np.flatnonzero(duplicated_batch["roles"] == "PlanningAgent").tolist()
    first_workflow_rows = [
        row for row in planner_rows if duplicated_batch["trajectory_ids"][row] == "w0"
    ]
    combined = duplicated.coefficients[first_workflow_rows].sum(dim=0)
    base_w0 = np.flatnonzero(
        (make_batch()["roles"] == "PlanningAgent")
        & (make_batch()["trajectory_ids"] == "w0")
    )[0]
    assert torch.allclose(combined, base.coefficients[base_w0], atol=1e-7, rtol=0.0)
    assert duplicated.metrics["aux/planner_outcome_contrast/duplicate_planner_row_count"] == 1.0


def test_explicit_padding_rows_are_ignored_instead_of_becoming_actions():
    batch = make_batch(duplicate_first=True)
    valid_rows = torch.ones(len(batch["roles"]), dtype=torch.bool)
    valid_rows[1] = False

    padded = build_planner_outcome_contrast_coefficients(
        **batch, valid_rows=valid_rows
    )
    base = build_planner_outcome_contrast_coefficients(**make_batch())

    assert torch.count_nonzero(padded.coefficients[1]) == 0
    assert padded.metrics["aux/planner_outcome_contrast/ignored_padding_rows"] == 1.0
    remaining = torch.cat((padded.coefficients[:1], padded.coefficients[2:]), dim=0)
    torch.testing.assert_close(remaining, base.coefficients)


def test_data_parallel_scaling_compensates_gradient_average():
    single = build_planner_outcome_contrast_coefficients(**make_batch(), data_parallel_size=1)
    parallel = build_planner_outcome_contrast_coefficients(**make_batch(), data_parallel_size=2)
    assert torch.allclose(parallel.coefficients / 2, single.coefficients)


def test_shuffled_control_changes_labels_within_question_and_preserves_pair_count():
    batch = make_batch()
    real = build_planner_outcome_contrast_coefficients(**batch)
    shuffled = build_planner_outcome_contrast_coefficients(
        **batch, shuffled=True, shuffle_seed=17
    )
    original = {f"w{index}": value for index, value in enumerate((1.0, 0.0, 1.0, 0.0))}
    assert shuffled.assigned_outcomes != original
    assert sorted(shuffled.assigned_outcomes.values()) == sorted(original.values())
    assert shuffled.metrics["aux/planner_outcome_contrast/pair_count"] == real.metrics[
        "aux/planner_outcome_contrast/pair_count"
    ]
    assert shuffled.metrics["aux/planner_outcome_contrast/shuffled_label_count"] >= 2.0


def test_rejects_multiple_logical_planner_actions_in_one_workflow():
    batch = make_batch()
    planner_row = int(np.flatnonzero(batch["roles"] == "PlanningAgent")[0])
    batch["workflow_steps"][planner_row] = 9
    extra = {key: value.copy() if isinstance(value, np.ndarray) else value.clone() for key, value in batch.items()}
    for key in ("response_mask", "old_log_probs"):
        extra[key] = torch.cat([extra[key], extra[key][planner_row : planner_row + 1]], dim=0)
    for key in ("question_ids", "trajectory_ids", "roles", "outcomes", "workflow_steps"):
        extra[key] = np.concatenate([extra[key], extra[key][planner_row : planner_row + 1]])
    extra["workflow_steps"][-1] = 10
    with pytest.raises(ValueError, match="logical Planner actions"):
        build_planner_outcome_contrast_coefficients(**extra)


def test_parameter_gradient_audit_summary_uses_global_inner_products():
    summary = summarize_parameter_gradient_audit(
        em_squared_norm=25.0,
        real_squared_norm=9.0,
        shuffled_squared_norm=4.0,
        real_em_dot=12.0,
        shuffled_em_dot=-5.0,
        real_shuffled_dot=3.0,
        loss_weight=0.1,
    )

    assert summary[
        "aux/planner_outcome_contrast/parameter_em_gradient_norm"
    ] == pytest.approx(5.0)
    assert summary[
        "aux/planner_outcome_contrast/parameter_weighted_real_poc_to_em_gradient_ratio"
    ] == pytest.approx(0.06)
    assert summary[
        "aux/planner_outcome_contrast/parameter_real_poc_em_cosine"
    ] == pytest.approx(0.8)
    assert summary[
        "aux/planner_outcome_contrast/parameter_shuffled_poc_em_cosine"
    ] == pytest.approx(-0.5)
    assert summary[
        "aux/planner_outcome_contrast/parameter_real_shuffled_cosine"
    ] == pytest.approx(0.5)
