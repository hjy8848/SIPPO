from pathlib import Path


ROOT = Path(__file__).parents[2]


def test_training_has_no_classification_rollout():
    trainer = (ROOT / "verl/trainer/ppo/ray_trainer.py").read_text(encoding="utf-8")

    assert 'trans_rawprompt_to_ids("ClassificationAgent"' not in trainer
    assert "agent_name='ClassificationAgent'" not in trainer
    assert "parse_response(\"ClassificationAgent\"" not in trainer


def test_eval_calls_untyped_planner():
    server = (ROOT / "exp/rosa_server.py").read_text(encoding="utf-8")
    start = server.index("async def generate_plan")
    end = server.index("async def generate_two_stage_plan")
    generate_plan = server[start:end]

    assert "await self.generate_plan(question)" in server
    assert "active_planner_system_message(" in generate_plan
    assert '"role": "system", "content": planner_system_msg' in generate_plan
    assert "<type>" not in generate_plan
    assert "topology" not in generate_plan


def test_eval_separates_prompt_budget_from_runtime_truncation():
    server = (ROOT / "exp/rosa_server.py").read_text(encoding="utf-8")

    assert 'os.environ.get("PLANNER_PROMPT_MAX_NODES", "5")' in server
    assert "build_planner_system_message(PLANNER_PROMPT_MAX_NODES)" in server
    assert "if PLANNER_PROMPT_MAX_NODES > 0" in server
    assert "with at most {MAX_NODES_PER_TURN} nodes" not in server


def test_eval_source_has_no_legacy_plan_classifier():
    server = (ROOT / "exp/rosa_server.py").read_text(encoding="utf-8")

    for legacy_symbol in (
        "ClassificationAgent",
        "VALID_PLAN_TYPES",
        "extract_plan_type",
        "validate_plan_shape",
        "heuristic_type",
        "PLANNER_PROMPTS",
    ):
        assert legacy_symbol not in server


def test_eval_trace_exposes_exact_sft_events():
    server = (ROOT / "exp/rosa_server.py").read_text(encoding="utf-8")

    assert "self._record_sft_event(" in server
    assert '"PlanningAgent",' in server
    assert '"AnswerGenerationAgent"' in server
    assert '"ReasonerAgent"' in server
    assert '"AnswerSummarizationAgent"' in server
    assert '"sft_events": self.sft_events' in server


def test_eval_supports_optional_planner_specialist_with_main_fallback():
    server = (ROOT / "exp/rosa_server.py").read_text(encoding="utf-8")

    assert 'PLANNER_API_URL = os.environ.get("PLANNER_API_URL", "").strip()' in server
    assert 'PLANNER_FALLBACK_TO_MAIN = env_flag("PLANNER_FALLBACK_TO_MAIN", "1")' in server
    assert 'return output, "main"' in server
    assert 'return content.strip(), "specialist"' in server
    assert "await self.generate_plan(question, force_main=True)" in server
    assert '"planner_route": planner_route' in server


def test_eval_supports_frozen_causal_exact_routing():
    server = (ROOT / "exp/rosa_server.py").read_text(encoding="utf-8")

    assert 'os.environ.get("PLANNER_ROUTE_MODE", "all")' in server
    assert "classify_causal_comparison_question(question)" in server
    assert "not is_exact_causal_plan(nodes)" in server
    assert 'route_decision["reasoning_mode"] == "anchor_distance"' in server
    assert 'planning_result["planner_route"] == "specialist"' in server


def test_eval_can_route_an_accepted_specialist_workflow_without_changing_default():
    server = (ROOT / "exp/rosa_server.py").read_text(encoding="utf-8")

    assert 'os.environ.get("SPECIALIST_ROUTE_SCOPE", "planner")' in server
    assert 'SPECIALIST_ROUTE_SCOPE == "workflow"' in server
    assert 'planner_routing["specialist_accepted"]' in server
    assert 'planner_routing["workflow_specialist_active"]' in server
    assert server.count("use_specialist=specialist_workflow_active") >= 7


def test_eval_supports_explicit_frozen_plan_override_for_mechanism_ablation():
    server = (ROOT / "exp/rosa_server.py").read_text(encoding="utf-8")
    evaluator = (ROOT / "exp/eval_rosa_with_trace.py").read_text(encoding="utf-8")

    assert "plan_override: Optional[str] = None" in server
    assert "if self.plan_override:" in server
    assert '"plan_override_used": bool(self.plan_override)' in server
    assert 'item.get("gold_plan")' in evaluator
    assert 'payload["plan_override"]' in evaluator


def test_eval_supports_frozen_raw_planner_output_before_system_compiler():
    server = (ROOT / "exp/rosa_server.py").read_text(encoding="utf-8")

    assert "planner_output_override: Optional[str] = None" in server
    assert 'planner_route = "planner_output_override"' in server
    assert "raw_model_plan_output = plan_output" in server
    assert "compiler(question," not in server
    assert "question,\n                        raw_model_plan_output," in server
    assert '"planner_output_override_used": bool(self.planner_output_override)' in server
    assert "and not self.planner_output_override" in server
    assert "plan_override and planner_output_override are mutually exclusive" in server
