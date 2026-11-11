import pytest
import torch

from verl.trainer.ppo.nrpsc_objective import (
    PlanObservation, PlanView, PreferencePair, actual_reward_eligibility, align_nodes, batch_node_coefficients, compare_plans, construct_node_pairs, frozen_tfidf, global_question_scale, node_preference_loss, node_preference_coefficients, node_token_mask, optional_psc_metadata, parse_plan, planner_node_masks, planner_plan_view, select_replay_rows,
    reward_eligibility_mask, reward_eligible, shared_prefix_masks,
)


def test_only_different_nodes_receive_gradients():
    scores = torch.tensor([[-1.0, -2.0, -3.0], [-1.0, -4.0, -5.0]], requires_grad=True)
    mask = torch.tensor([False, True, True])
    pair = PreferencePair('q', 0, 1, (mask,), (mask,))
    node_preference_loss(scores, [pair], question_count=1).backward()
    assert scores.grad[0, 0] == scores.grad[1, 0] == 0
    assert scores.grad[0, 1] < 0 < scores.grad[1, 1]


def test_all_question_normalization_prevents_single_group_amplification():
    scores = torch.zeros(2, 2, requires_grad=True)
    mask = torch.tensor([True, False])
    pair = PreferencePair('q', 0, 1, (mask,), (mask,))
    one = node_preference_loss(scores, [pair], question_count=1)
    four = node_preference_loss(scores, [pair], question_count=4)
    assert torch.allclose(one, four * 4)


def test_shared_prefix_and_reward_eligibility():
    a, b = shared_prefix_masks(torch.tensor([1, 2, 3]), torch.tensor([1, 2, 4, 5]))
    assert a.tolist() == [False, False, True]
    assert b.tolist() == [False, False, True, True]
    assert reward_eligible(1, 1, True)
    assert not reward_eligible(1, 0, True)
    assert not reward_eligible(1, 1, False)


def test_empty_or_invalid_decisions_are_rejected():
    scores = torch.zeros(2, 3)
    with pytest.raises(ValueError):
        node_preference_loss(scores, [PreferencePair('q', 0, 1, (), ())], question_count=1)


def test_one_sided_gradient_changes_only_the_provided_decision():
    scores = torch.zeros(2, 3, requires_grad=True)
    mask = torch.tensor([True, False, False])
    loss = node_preference_loss(scores, [PreferencePair('q', 0, 1, (), (mask,))], question_count=1)
    loss.backward()
    assert scores.grad[0].abs().sum() == 0
    assert scores.grad[1, 0] == 0.5


def test_distributed_question_scale_and_binary_eligibility():
    assert global_question_scale(1, 4, 2) == 0.5
    assert global_question_scale(3, 4, 2) == 1.5
    assert global_question_scale(0, 0, 2) == 0.0
    mask = reward_eligibility_mask(torch.tensor([1, 1, 0]), torch.tensor([1, 0, 0]),
                                   torch.tensor([True, True, False]))
    assert mask.tolist() == [True, False, True]
    with pytest.raises(ValueError):
        reward_eligibility_mask(torch.tensor([1]), torch.tensor([0.5]), torch.tensor([True]))


def test_node_offsets_and_boundary_crossing():
    assert node_token_mask([(0, 2), (2, 5), (5, 8)], 2, 5).tolist() == [False, True, False]
    assert node_token_mask([(0, 3), (3, 6), (6, 8)], 2, 5).tolist() == [False, True, False]


def test_dag_alignment_respects_edges_and_node_type():
    a = parse_plan('<search id="1">who is X</search><reason id="2">is #1# American</reason>')
    b = parse_plan('<search id="4">who is X</search><reason id="9">is #4# American</reason>')
    scorer = frozen_tfidf([n.text for n in a + b])
    assert align_nodes(a, b, scorer, 0.8) == ((0, 0), (1, 1))
    with pytest.raises(ValueError, match="unresolved"):
        parse_plan('<reason id="1">is #2# American</reason>')


def test_optional_metadata_keeps_eligibility_when_present():
    assert optional_psc_metadata({'psc_chosen_eligible': [False], 'uid': ['q']}) == ['psc_chosen_eligible']
    assert optional_psc_metadata({'uid': ['q']}) == []


def test_planner_node_masks_validate_token_roundtrip():
    text = '<search id="1">find X</search>'
    class CharacterTokenizer:
        def decode(self, ids, **kwargs):
            return ''.join(map(chr, ids))
        def __call__(self, value, **kwargs):
            return {'input_ids': list(map(ord, value)),
                    'offset_mapping': [(i, i + 1) for i in range(len(value))]}
    nodes, masks = planner_node_masks(CharacterTokenizer(), list(map(ord, text)))
    assert len(nodes) == len(masks) == 1 and masks[0].all()
    class BrokenTokenizer(CharacterTokenizer):
        def __call__(self, value, **kwargs):
            result = super().__call__(value, **kwargs)
            result['input_ids'][-1] = 0
            return result
    with pytest.raises(ValueError, match='round-trip'):
        planner_node_masks(BrokenTokenizer(), list(map(ord, text)))


def test_linearized_coefficients_equal_direct_pairwise_gradient():
    values = torch.tensor([[-1., -2., -3.], [-2., -1., -4.]], requires_grad=True)
    first = torch.tensor([False, True, False])
    second = torch.tensor([False, False, True])
    pairs = [PreferencePair('q', 0, 1, (first, second), (first,), 0.7)]
    loss = node_preference_loss(values, pairs, question_count=3)
    direct = torch.autograd.grad(loss, values)[0]
    coeffs = node_preference_coefficients(values, pairs, question_count=3)
    assert torch.allclose(direct, coeffs)
    assert torch.allclose(torch.autograd.grad((values * coeffs).sum(), values)[0], direct)


def test_reward_eligibility_intersects_existing_and_rejects_missing():
    metadata = {'outcome_exact': [1, 1, 0], 'outcome_reward': [1, 0, 0],
                'workflow_valid': [True, True, False], 'psc_chosen_eligible': [False, True, True]}
    assert actual_reward_eligibility(metadata).tolist() == [False, False, True]
    del metadata['outcome_reward']
    with pytest.raises(ValueError, match='missing'):
        actual_reward_eligibility(metadata)


def test_subset_requires_exact_prefix_before_end_marker():
    short_nodes = parse_plan('<search id="1">find X</search>')
    long_nodes = parse_plan('<search id="1">find X</search><search id="2">find Y</search>')
    short = PlanView(short_nodes, (torch.tensor([True, False, False]),), (9,), (1, 2, 0),
                     torch.tensor([False, False, True]))
    long = PlanView(long_nodes, (torch.tensor([True, False, False]), torch.tensor([False, False, True])),
                    (9,), (1, 2, 3), torch.tensor([False, False, True]))
    score = lambda a, b: 1.0 if a == b else 0.0
    aligned = compare_plans(long, short, score, 1.0)
    assert aligned.kind == 'S' and len(aligned.left_nodes) == len(aligned.right_nodes) == 1
    mismatch = PlanView(short_nodes, short.node_masks, (8,), short.response_ids, short.end_mask)
    assert compare_plans(long, mismatch, score, 1.0).kind == 'N'


def test_class_grouping_uses_all_terminal_observations_and_calibration():
    mask = torch.tensor([True, False])
    end = torch.tensor([False, True])
    a = PlanView(parse_plan('<search id="1">find X</search>'), (mask,), (9,), (1, 0), end)
    b = PlanView(parse_plan('<search id="1">find Y</search>'), (mask,), (9,), (2, 0), end)
    observations = [PlanObservation('q', 0, a, 1, 1), PlanObservation('q', 1, a, 1, 1),
                    PlanObservation('q', 2, b, 0, 2)]
    score = lambda x, y: float(x == y)
    pairs, counts = construct_node_pairs(observations, score, 1., lambda s, n, t, m: 0.75)
    assert counts['classes'] == 2 and counts['informative_class_pairs'] == 1
    assert len(pairs) == 2 and all(pair.weight == 0.75 for pair in pairs)
    assert all(pair.winner in (0, 1) and pair.loser == 2 for pair in pairs)


def test_plan_view_includes_closing_marker_and_preserves_prompt():
    text = '<plan><search id="1">X</search></plan>'
    class Tokenizer:
        eos_token_id = None
        def decode(self, ids, **kwargs):
            return ''.join(map(chr, ids))
        def __call__(self, value, **kwargs):
            return {'input_ids': list(map(ord, value)),
                    'offset_mapping': [(i, i + 1) for i in range(len(value))]}
    view = planner_plan_view(Tokenizer(), [17], list(map(ord, text)))
    assert view.prompt_ids == (17,)
    assert int(view.end_mask.sum()) == len('</plan>')
    assert not (view.end_mask & view.node_masks[0]).any()


def test_batch_coefficients_ignore_padding_and_nonplanner_rows():
    mask = torch.tensor([True, False])
    end = torch.tensor([False, True])
    a = PlanView(parse_plan('<search id="1">X</search>'), (mask,), (9,), (1, 0), end)
    b = PlanView(parse_plan('<search id="1">Y</search>'), (mask,), (9,), (2, 0), end)
    obs = [PlanObservation('q', 0, a, 1, 1), PlanObservation('q', 2, b, 0, 1)]
    values = torch.zeros(3, 4)
    response_mask = torch.tensor([[0, 1, 1, 0], [1, 1, 0, 0], [1, 0, 1, 0]])
    coeffs, counts = batch_node_coefficients(values, response_mask, obs, lambda x, y: float(x == y),
                                              1.0, question_count=1)
    assert counts['G'] == 1
    assert coeffs[0, 1] < 0 < coeffs[2, 0]
    assert coeffs[1].abs().sum() == 0
    assert coeffs[response_mask == 0].abs().sum() == 0


def test_replay_budget_prioritizes_mixed_questions_and_distinct_classes():
    mask, end = torch.tensor([True, False]), torch.tensor([False, True])
    a = PlanView(parse_plan('<search id="1">X</search>'), (mask,), (9,), (1, 0), end)
    b = PlanView(parse_plan('<search id="1">Y</search>'), (mask,), (9,), (2, 0), end)
    obs = [PlanObservation('uniform', 0, a, 0, 1), PlanObservation('uniform', 1, b, 0, 1),
           PlanObservation('mixed', 2, a, 1, 1), PlanObservation('mixed', 3, b, 0, 1)]
    assert select_replay_rows(obs, lambda x, y: float(x == y), 1., 3) == [2, 3, 0]
