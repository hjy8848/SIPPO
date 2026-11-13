"""Subgoal Plan v1 (qa_manager.plan_format=subgoal): parser, execution-time routing, final-answer stage, NR-PSC parsing.

CPU only. The model is a scripted stub keyed on each prompt; parsing uses the real ROSA_AgentManager methods.
"""
import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from qa_manager.ROSA import protocol as P
from qa_manager.ROSA.rosa_agent_manager import ROSA_AgentManager
from sippo.workflow_rollout import WorkflowRolloutMixin
from verl import DataProto
from verl.trainer.ppo import nrpsc_objective as N


class ScriptedQA(ROSA_AgentManager):
    """Real prompt building and parsing; tokenization, retrieval and decoding are stubbed."""

    def __init__(self, plan_format="subgoal"):
        self.plan_format = plan_format
        self.contract_preserving_dag_v2 = False
        self.max_nodes_per_turn = 0
        self.planner_prompt_max_nodes = 0
        self.Prompts = {
            "PlanningAgent": P.build_subgoal_planner_system_message(0),
            "QueryRewriteAgent": P.QUERY_REWRITE_SYSTEM_MSG,
            "AnswerGenerationAgent": P.ANSWER_GENERATION_SYSTEM_MSG,
            "ReasonerAgent": P.REASON_SYSTEM_MSG,
            "AnswerSummarizationAgent": P.SUBGOAL_FINAL_SYSTEM_MSG,
        }
        self.retrieved = []
        self.prompts = []

    def _messages_to_ids(self, messages):
        # The rollout keeps only keys present in the real batch, so the prompt travels as an index in input_ids.
        self.prompts.append(messages)
        return {"raw_prompt": messages, "input_ids": torch.tensor([len(self.prompts) - 1, 0, 0])}

    def retrieval(self, query, top_k=None):
        self.retrieved.append(query)
        return [f"Doc about {query}"]

    def get_answers_text(self, data):
        return [str(data.non_tensor_batch["fake_text"][0])]


CFG = OmegaConf.create({
    "algorithm": {"adv_estimator": "workflow_rloo"},
    "actor_rollout_ref": {"rollout": {"n": 1, "free_cache_engine": False}, "actor": {}},
    "qa_manager": {"result_reward_only": True, "workflow_rollout_n": 2, "final_reward_em_weight": 1.0,
                   "retrieval_topk": 1, "max_nodes_per_turn": 0, "search_rewrite_max_retries": 1,
                   "invalid_workflow_reward_cap": 0.0, "plan_format": "subgoal"},
})


def make_runner(script):
    """``script(system, user) -> text`` decides every model output; returns (trainer, qa, calls)."""
    calls = []
    qa_holder = []

    class T(WorkflowRolloutMixin):
        pass

    t = T()
    t.config = CFG
    t.global_steps = 0

    def fake_inference(input_dict, agent_name="DefaultAgent", sampling_seed=None, sampling_temperature=None):
        messages = qa_holder[0].prompts[int(input_dict["input_ids"].reshape(-1)[0])]
        text = script(messages[0]["content"], messages[-1]["content"])
        calls.append((agent_name, messages[0]["content"][:30], text))
        prompts = torch.ones(1, 3, dtype=torch.long)
        responses = torch.arange(4).view(1, -1)
        attn = torch.ones(1, 7, dtype=torch.long)
        dp = DataProto.from_single_dict({"prompts": prompts, "responses": responses, "attention_mask": attn,
                                         "response_mask": attn[:, 3:], "input_ids": torch.cat([prompts, responses], 1)})
        dp.non_tensor_batch["fake_text"] = np.array([text], dtype=object)
        return dp

    t.async_inference = fake_inference
    t._check_slim_rollout_config()
    qa = ScriptedQA()
    qa_holder.append(qa)
    return t, qa, calls


def run(script, question, answer):
    t, qa, calls = make_runner(script)
    target = {"input_ids": torch.ones(1, 3, dtype=torch.long), "data_source": np.array(["hotpotqa"], dtype=object)}
    rows, metrics = t.async_rollout_single(qa, target, question, answer, False)
    return rows, metrics, qa, calls


def script_from(plan, route=None, ag=None, final=None):
    route, ag = route or {}, ag or {}

    def script(system, user):
        if system.startswith("Decompose the question"):
            return plan
        if system == P.SUBGOAL_ROUTE_SYSTEM_MSG:
            for key, out in route.items():
                if key in user:
                    return out
            return "<need_search/>"
        if system == P.QUERY_REWRITE_SYSTEM_MSG:  # substitute the "# k = value" bindings into the draft
            lines = user.splitlines()
            draft = lines[1].removeprefix("Draft: ")
            for line in lines[2:]:
                if line.startswith("# ") and " = " in line:
                    ref, value = line[2:].split(" = ", 1)
                    draft = draft.replace(f"#{ref}#", value)
            return draft
        if system == P.ANSWER_GENERATION_SYSTEM_MSG:
            for key, out in ag.items():
                if key in user:
                    return out
            return "<conclusion>Insufficient information</conclusion>"
        if system == P.SUBGOAL_FINAL_SYSTEM_MSG:
            return final or "<answer>" + user.split("**Final Sub-question Answer**: ")[1].strip() + "</answer>"
        if system == P.FINAL_SYSTEM_MSG:
            return "<answer>fallback guess</answer>"
        raise AssertionError(f"unexpected prompt: {system[:60]}")

    return script


# ---------------------------------------------------------------- parser
@pytest.mark.parametrize("plan,ok,reason", [
    ('<plan>\n<step id="1">Who was Ada Lovelace\'s husband?</step>\n<step id="2">Who was the father of #1#?</step>\n</plan>', True, None),
    ('<plan><step id="1">When was Alan Turing born?</step><step id="2">When was von Neumann born?</step>'
     '<step id="3">Who was born earlier, #1# or #2#?</step></plan>', True, None),
    ('<plan><search id="1">Who was Ada Lovelace\'s husband?</search></plan>', False, "no_valid_nodes"),
    ('<plan><step id="1">Who was Ada Lovelace\'s husband?</step><reason id="2">#1#</reason></plan>', False, "illegal_tags"),
    ('<plan><step id="1">Who was Ada Lovelace\'s husband?</step> and then <step id="2">father of #1#?</step></plan>', False, "text_outside_steps"),
    ('<plan><step id="2">Who was Ada Lovelace\'s husband?</step></plan>', False, "non_contiguous_node_ids"),
    ('<plan><step id="1">Who was the father of #2#?</step><step id="2">Who was Ada Lovelace\'s husband?</step></plan>', False, "future_or_self_reference"),
    ('<plan><step id="1">What is the capital of Peru?</step></plan>', False, "invalid_plan"),  # root step not grounded
    ('<plan><step id="1">{CONCRETE_SUBQUESTION}</step></plan>', False, "invalid_plan"),
    ('Sure! <plan><step id="1">Who was Ada Lovelace\'s husband?</step></plan>', False, "missing_plan_tag"),
])
def test_subgoal_parser(plan, ok, reason):
    question = ("Who was born earlier, Alan Turing or John von Neumann?" if "Turing" in plan
                else "Who is the father of Ada Lovelace's husband?")
    out = ScriptedQA().parse_response("PlanningAgent", plan, question=question)
    assert out["is_valid"] is ok, out
    if ok:
        assert all(node["type"] == "step" for node in out["nodes"])
    else:
        assert out["reason"] == reason, out


def test_typed_parser_unchanged_and_rejects_steps():
    typed = ScriptedQA(plan_format="typed")
    q = "Who is the father of Ada Lovelace's husband?"
    good = '<plan><search id="1">Who was Ada Lovelace\'s husband?</search><search id="2">Who was the father of #1#?</search></plan>'
    assert typed.parse_response("PlanningAgent", good, question=q)["is_valid"]
    steps = '<plan><step id="1">Who was Ada Lovelace\'s husband?</step></plan>'
    assert not typed.parse_response("PlanningAgent", steps, question=q)["is_valid"]


def test_planner_prompt_switches_with_plan_format():
    msgs = ScriptedQA().trans_rawprompt_to_ids("PlanningAgent", "Q?")["raw_prompt"]
    assert "<step id=" in msgs[0]["content"] and "<search" not in msgs[0]["content"]
    typed = ScriptedQA(plan_format="typed").trans_rawprompt_to_ids("PlanningAgent", "Q?")["raw_prompt"]
    assert "<search id=" in typed[0]["content"]


@pytest.mark.parametrize("text,route,valid", [
    ("<need_search/>", "search", True), ("<need_search>", "search", True), (" <need_search></need_search> ", "search", True),
    ("<conclusion>Alan Turing</conclusion>", "memory", True),
    ("<conclusion>Insufficient information</conclusion>", "search", True),
    ("Alan Turing", "search", False), ("<think>x</think><conclusion>A</conclusion>", "search", False),
])
def test_route_parser(text, route, valid):
    out = P.parse_subgoal_route(text)
    assert (out["route"], out["is_valid"]) == (route, valid)


# ---------------------------------------------------------------- execution
def test_bridge_chain_searches_both_steps():
    plan = '<plan><step id="1">Who was Ada Lovelace\'s husband?</step><step id="2">Who was the father of #1#?</step></plan>'
    rows, m, qa, calls = run(script_from(plan, ag={"husband": "<conclusion>William King</conclusion>",
                                                   "father of William King": "<conclusion>Peter King</conclusion>"}),
                             "Who is the father of Ada Lovelace's husband?", "Peter King")
    assert m["final_em"] == 1.0 and m["final_answer"] == "Peter King"
    assert (m["subgoal_steps"], m["subgoal_memory_steps"], m["subgoal_search_steps"], m["summary_fallback"]) == (2, 0, 2, 0.0)
    assert m["retrieval_times"] == 2 and qa.retrieved == ["Who was Ada Lovelace's husband?", "Who was the father of William King?"]
    node_types = [str(r.non_tensor_batch["node_type"][0]) for r in rows]
    assert node_types == ["planning", "search_answer", "route", "query_rewrite", "search_answer", "summary"]
    assert {float(r.non_tensor_batch["outcome_reward"][0]) for r in rows} == {1.0}


def test_comparison_answers_last_step_from_memory():
    plan = ('<plan><step id="1">When was Alan Turing born?</step><step id="2">When was John von Neumann born?</step>'
            '<step id="3">Who was born earlier, Alan Turing (born #1#) or John von Neumann (born #2#)?</step></plan>')
    rows, m, qa, calls = run(script_from(plan, route={"born earlier": "<conclusion>John von Neumann</conclusion>"},
                                         ag={"Alan Turing born": "<conclusion>23 June 1912</conclusion>",
                                             "von Neumann born": "<conclusion>28 December 1903</conclusion>"}),
                             "Who was born earlier, Alan Turing or John von Neumann?", "John von Neumann")
    assert m["final_em"] == 1.0
    assert (m["subgoal_memory_steps"], m["subgoal_search_steps"], m["retrieval_times"]) == (1, 2, 2)
    route_prompt = [c for c in calls if c[0] == "ReasonerAgent"]
    assert len(route_prompt) == 1
    final_call = calls[-1]
    assert final_call[1].startswith("Give the final answer") and m["summary_fallback"] == 0.0


def test_failed_final_step_falls_back_to_history_and_is_recorded():
    plan = '<plan><step id="1">Who was Ada Lovelace\'s husband?</step><step id="2">Who was the father of #1#?</step></plan>'
    rows, m, qa, calls = run(script_from(plan, ag={"husband": "<conclusion>William King</conclusion>"}),
                             "Who is the father of Ada Lovelace's husband?", "Peter King")
    assert m["summary_fallback"] == 1.0 and m["final_answer"] == "fallback guess"
    assert calls[-1][1].startswith("Synthesize the resolved")


def test_malformed_route_is_invalid_and_still_searched():
    plan = ('<plan><step id="1">When was Alan Turing born?</step><step id="2">When was John von Neumann born?</step>'
            '<step id="3">Who was born earlier, #1# or #2#?</step></plan>')
    rows, m, qa, calls = run(script_from(plan, route={"born earlier": "John von Neumann"},
                                         ag={"Alan Turing born": "<conclusion>1912</conclusion>",
                                             "von Neumann born": "<conclusion>1903</conclusion>",
                                             "earlier": "<conclusion>John von Neumann</conclusion>"}),
                             "Who was born earlier, Alan Turing or John von Neumann?", "John von Neumann")
    assert m["workflow_valid"] == 0.0 and m["final_reward"] == 0.0  # invalid action -> reward capped, as for typed plans
    assert (m["subgoal_memory_steps"], m["subgoal_search_steps"], m["retrieval_times"]) == (0, 3, 3)


def test_invalid_plan_returns_early_with_planner_row_only():
    rows, m, qa, calls = run(script_from('<plan><search id="1">x</search></plan>'), "Who is X?", "Y")
    assert len(rows) == 1 and m["planner_valid"] == 0.0 and m["final_reward"] == 0.0


def test_batch_metrics_include_subgoal_counters():
    plan = '<plan><step id="1">Who was Ada Lovelace\'s husband?</step><step id="2">Who was the father of #1#?</step></plan>'
    t, qa, _ = make_runner(script_from(plan, ag={"husband": "<conclusion>William King</conclusion>",
                                                 "father": "<conclusion>Peter King</conclusion>"}))
    batch = {"input_ids": torch.ones(2, 3, dtype=torch.long),
             "extra_info": np.array([{"question": "Who is the father of Ada Lovelace's husband?", "answer": ["Peter King"]}] * 2, dtype=object),
             "data_source": np.array(["hotpotqa"] * 2, dtype=object)}
    rows, metrics = t.async_rollout_batch(batch, qa, parallel=False)
    assert metrics["train/subgoal/steps_mean"] == 2.0 and metrics["train/subgoal/search_steps_mean"] == 2.0
    assert metrics["train/subgoal/memory_step_share"] == 0.0 and metrics["train/subgoal/summary_fallback_rate"] == 0.0
    assert metrics["train/rollout/em_aggregate"] == 1.0


def test_typed_config_rejects_unknown_plan_format():
    t, _, _ = make_runner(script_from(""))
    t.config = OmegaConf.merge(CFG, {"qa_manager": {"plan_format": "free"}})
    with pytest.raises(ValueError):
        t._check_slim_rollout_config()


# ---------------------------------------------------------------- NR-PSC
def test_nrpsc_parses_and_aligns_step_plans():
    a = N.parse_plan('<plan><step id="1">When was Alan Turing born?</step><step id="2">Who was born earlier, #1# or X?</step></plan>')
    b = N.parse_plan('<plan><step id="1">When was Alan Turing born?</step><step id="2">Who was born first, #1# or X?</step></plan>')
    assert [n.kind for n in a] == ["step", "step"] and a[1].references == ("1",)
    sim = N.frozen_tfidf([n.text for n in a + b])
    assert N.align_nodes(a, b, sim, 0.5) == ((0, 0), (1, 1))
    typed = N.parse_plan('<plan><search id="1">When was Alan Turing born?</search></plan>')
    assert N.align_nodes(a[:1], typed, sim, 0.0) == ()  # step and search nodes never align
    with pytest.raises(ValueError):
        N.parse_plan('<plan><step id="1">x</step><step id="2">y</plan>')
