from pathlib import Path


ROOT = Path(__file__).parents[2]


def test_accuracy_first_uses_complete_workflow_rloo():
    script = (ROOT / "run_rosa_train_accuracy_first.sh").read_text(encoding="utf-8")

    assert "export RESULT_REWARD_ONLY=true" in script
    assert "export ADV_ESTIMATOR=workflow_rloo" in script
    assert 'WORKFLOW_ROLLOUT_N="${WORKFLOW_ROLLOUT_N:-4}"' in script
    assert 'USE_KL_LOSS="${USE_KL_LOSS:-true}"' in script


def test_main_launcher_forwards_accuracy_first_settings():
    script = (ROOT / "run_rosa_train_2x_rtx_pro_6000.sh").read_text(encoding="utf-8")

    required_overrides = [
        'algorithm.adv_estimator="${ADV_ESTIMATOR}"',
        'actor_rollout_ref.actor.use_kl_loss="${USE_KL_LOSS}"',
        '+qa_manager.workflow_rollout_n="${WORKFLOW_ROLLOUT_N}"',
        '+qa_manager.final_reward_em_weight="${FINAL_REWARD_EM_WEIGHT}"',
        '+qa_manager.retrieval_topk="${RETRIEVAL_TOPK}"',
    ]
    for override in required_overrides:
        assert override in script
