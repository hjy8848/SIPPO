from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_online_launcher_starts_from_sft_with_clean_em_grpo_data():
    script = (
        ROOT / "exp" / "run_dagent_online_opsd_grpo_elastic_a6000.sh"
    ).read_text(encoding="utf-8")

    assert "global_step_510/huggingface" in script
    assert "rl_train_mixed_2wiki_hotpot_seed42.parquet" in script
    assert "reason140" not in script
    assert "export ADV_ESTIMATOR=workflow_grpo" in script
    assert "export RESULT_REWARD_ONLY=true" in script
    assert "export FINAL_REWARD_EM_WEIGHT=1.0" in script
    assert "export DAG_OPSD_ENABLED=false" in script
    assert "export ONLINE_OPSD_ENABLED=true" in script
    assert "ONLINE_OPSD_UPDATE_INTERVAL:-4" in script
    assert "ONLINE_OPSD_REQUIRE_REASON_TO_SEARCH=false" in script
    assert "ONLINE_OPSD_REWRITE_AFTER_REASON_ONLY=false" in script
    assert "AnswerGenerationAgent" in script


def test_online_opsd_is_role_selective_and_not_an_advantage_estimator():
    trainer = (ROOT / "verl" / "trainer" / "ppo" / "ray_trainer.py").read_text(
        encoding="utf-8"
    )
    actor = (ROOT / "verl" / "workers" / "actor" / "dp_actor.py").read_text(
        encoding="utf-8"
    )

    assert "workflow_dag_opsd_grpo" not in (
        ROOT / "exp" / "run_dagent_online_opsd_grpo_elastic_a6000.sh"
    ).read_text(encoding="utf-8")
    assert "require_reason_to_search" in trainer
    assert "query_rewrite_after_reason_only" in trainer
    assert '"workflow_valid"' in trainer
    assert "opsd_round % len(successful)" in trainer
    assert "Online OPSD must not distill AnswerSummarizationAgent" in trainer
    assert "def _update_online_opsd" in actor
    assert "self.actor_optimizer.zero_grad()" in actor
    assert "loss.backward()" in actor
    assert "self._optimizer_step()" in actor


def test_online_opsd_has_confidence_and_divergence_gates():
    actor = (ROOT / "verl" / "workers" / "actor" / "dp_actor.py").read_text(
        encoding="utf-8"
    )

    assert "min_teacher_confidence" in actor
    assert "min_forward_kl" in actor
    assert "max_forward_kl" in actor
    assert "divergence_temperature" in actor
    assert "teacher_confidence" in actor
    assert "gate_weight" in actor
