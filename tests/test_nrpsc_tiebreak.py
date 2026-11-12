"""Tests for the NR-PSC EM-tie retrieval tie-break: construct_node_pairs() and the rollout-side signal."""
from sippo.workflow_rollout import nrpsc_tiebreak_signal
from verl.trainer.ppo.nrpsc_objective import PlanObservation, construct_node_pairs


class _CharTokenizer:
    eos_token_id = None

    def decode(self, ids, **kwargs):
        return "".join(map(chr, ids))

    def __call__(self, text, **kwargs):
        return {"input_ids": list(map(ord, text)), "offset_mapping": [(i, i + 1) for i in range(len(text))]}


def _plan_view(text):
    from verl.trainer.ppo.nrpsc_objective import planner_plan_view

    ids = list(map(ord, text))
    return planner_plan_view(_CharTokenizer(), [1, 2], ids)


def _similarity(a, b):
    return 1.0 if a == b else 0.0


def test_tiebreak_disabled_by_default_is_identical_to_old_behaviour():
    """tiebreak_weight=0.0 (the default) must reproduce the pre-tiebreak counts and pairs exactly."""
    plan_a = '<search id="1">alpha</search></plan>'
    plan_b = '<search id="1">beta</search></plan>'
    obs = [
        PlanObservation("q", 0, _plan_view(plan_a), 1, 1, True, secondary_successes=1, secondary_trials=1),
        PlanObservation("q", 1, _plan_view(plan_a), 1, 1, True, secondary_successes=0, secondary_trials=1),
        PlanObservation("q", 2, _plan_view(plan_b), 0, 1, True, secondary_successes=0, secondary_trials=1),
        PlanObservation("q", 3, _plan_view(plan_b), 0, 1, True, secondary_successes=1, secondary_trials=1),
    ]
    pairs, counts = construct_node_pairs(obs, _similarity, 0.8)  # tiebreak_weight defaults to 0.0
    assert counts["informative_class_pairs"] == 1
    assert counts["em_informative_class_pairs"] == 1
    assert counts["tiebreak_informative_class_pairs"] == 0
    assert len(pairs) > 0
    assert all(pair.weight == 1.0 for pair in pairs)


def test_em_tie_with_no_secondary_data_still_produces_nothing():
    plan_a = '<search id="1">alpha</search></plan>'
    plan_b = '<search id="1">beta</search></plan>'
    obs = [
        PlanObservation("q", 0, _plan_view(plan_a), 1, 1, True),
        PlanObservation("q", 1, _plan_view(plan_b), 1, 1, True),  # same EM (1/1 vs 1/1): a pure tie
    ]
    pairs, counts = construct_node_pairs(obs, _similarity, 0.8, tiebreak_weight=0.1)
    assert counts["class_pairs"] == 1
    assert counts["informative_class_pairs"] == 0  # no secondary data on either side -> skipped, as before
    assert pairs == []


def test_em_tie_broken_by_secondary_signal_when_enabled():
    plan_a = '<search id="1">alpha</search></plan>'
    plan_b = '<search id="1">beta</search></plan>'
    obs = [
        PlanObservation("q", 0, _plan_view(plan_a), 1, 1, True, secondary_successes=1, secondary_trials=1),
        PlanObservation("q", 1, _plan_view(plan_b), 1, 1, True, secondary_successes=0, secondary_trials=1),
    ]
    off_pairs, off_counts = construct_node_pairs(obs, _similarity, 0.8)  # default off
    assert off_pairs == [] and off_counts["informative_class_pairs"] == 0

    pairs, counts = construct_node_pairs(obs, _similarity, 0.8, tiebreak_weight=0.3)
    assert counts["tiebreak_informative_class_pairs"] == 1
    assert counts["em_informative_class_pairs"] == 0
    assert len(pairs) == 1
    assert pairs[0].winner == 0 and pairs[0].loser == 1  # plan_a has secondary_successes=1, wins the tie-break
    assert abs(pairs[0].weight - 0.3) < 1e-9  # scaled by tiebreak_weight (no reliability table here)


def test_em_difference_always_wins_over_secondary_signal():
    """An EM difference must never be overridden by the secondary signal, regardless of tiebreak_weight."""
    plan_a = '<search id="1">alpha</search></plan>'
    plan_b = '<search id="1">beta</search></plan>'
    obs = [
        PlanObservation("q", 0, _plan_view(plan_a), 0, 1, True, secondary_successes=1, secondary_trials=1),
        PlanObservation("q", 1, _plan_view(plan_b), 1, 1, True, secondary_successes=0, secondary_trials=1),
    ]
    pairs, counts = construct_node_pairs(obs, _similarity, 0.8, tiebreak_weight=1.0)
    assert counts["em_informative_class_pairs"] == 1
    assert counts["tiebreak_informative_class_pairs"] == 0
    assert len(pairs) == 1
    assert pairs[0].winner == 1 and pairs[0].loser == 0  # plan_b wins on EM, despite losing on the secondary signal
    assert pairs[0].weight == 1.0  # EM-based pairs are never scaled by tiebreak_weight


def test_negative_tiebreak_weight_rejected():
    plan_a = '<search id="1">alpha</search></plan>'
    obs = [PlanObservation("q", 0, _plan_view(plan_a), 1, 1, True, secondary_successes=1, secondary_trials=1)]
    try:
        construct_node_pairs(obs, _similarity, 0.8, tiebreak_weight=-0.1)
    except ValueError:
        return
    raise AssertionError("expected ValueError for a negative tiebreak_weight")


def test_rollout_signal_excludes_binary_and_question_substring_answers():
    intermediate = {"n1": {"docs": ["[1_1]: (Title: X) The capital of France is Paris, a large city."]}}
    usable, hit = nrpsc_tiebreak_signal("Is Paris the capital of France?", ["yes"], intermediate)
    assert usable is False  # binary answer: never usable

    usable, hit = nrpsc_tiebreak_signal("What is the capital of France called Paris?", ["Paris"], intermediate)
    assert usable is False  # answer string is already inside the question text

    usable, hit = nrpsc_tiebreak_signal("What is the capital of France?", ["Paris"], intermediate)
    assert usable is True and hit is True

    usable, hit = nrpsc_tiebreak_signal("What is the capital of France?", ["Berlin"], intermediate)
    assert usable is True and hit is False


def test_rollout_signal_handles_no_retrieval():
    usable, hit = nrpsc_tiebreak_signal("What is the capital of France?", ["Paris"], {})
    assert usable is True and hit is False
