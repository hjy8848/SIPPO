"""Translate a historical JADE launch command (W or P arm) into overrides for the verl 0.9.1 based SIPPO entry point.

usage: build_args.py HIST.cmd OUT_args.sh RUN_DIR EXP_NAME [--memory-safe] [--steps N] [--npu]
Prints a report of every change; nothing is changed silently.
"""
from __future__ import annotations

import shlex
import sys
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

TREE = str(Path(__file__).resolve().parents[1])  # the tree this script lives in (NR-PSC copy), never the original NPU tree
NRPSC_IDF_PATH = "/home/hejingyu/projects/JADE/rebuttal_audits/nrpsc_idf_canonical_sft_train_20261112_r2/idf.json"
NRPSC_IDF_SHA256 = "19a88a0a9a748c8c7e8b521b8eb6e093b7ff2cd964c275ae8c204f43d196c56b"
FAST = {  # historical W/P concurrency (JADE launch cmds), restored on top of the memory-safe profile for speed
    "qa_manager.max_workflow_concurrency": "64",
    "qa_manager.max_retrieval_concurrency": "32",
    "actor_rollout_ref.rollout.max_num_seqs": "64",
    "actor_rollout_ref.rollout.gpu_memory_utilization": "0.30",
}
DROP_PREFIXES = (
    "actor_rollout_ref.actor.online_opsd.",  # feature is off in W/P; the SIPPO updater rejects it if enabled
    "critic.",                                # workflow_rloo trains without a critic
)
DROP_KEYS: set[str] = set()  # (rollout.name is a mandatory key in 0.9.1 and must stay)
MAP_KEYS = {
    # JADE fused AdamW  ->  0.9.1 expresses optimizer kwargs through override_optimizer_config
    "actor_rollout_ref.actor.optim.fused": ("actor_rollout_ref.actor.optim.override_optimizer_config", "{fused:true}"),
}
MEMORY_SAFE = {  # values from the historical single-GPU P resume segment, for a 61 GB card
    "actor_rollout_ref.model.enable_gradient_checkpointing": "true",
    "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu": "1",
    "actor_rollout_ref.actor.fsdp_config.optimizer_offload": "true",
    "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu": "1",
    "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu": "1",
    "actor_rollout_ref.rollout.gpu_memory_utilization": "0.25",
    "actor_rollout_ref.rollout.max_num_batched_tokens": "16384",
    "actor_rollout_ref.rollout.max_num_seqs": "32",
    "qa_manager.max_retrieval_concurrency": "16",
    "qa_manager.max_workflow_concurrency": "32",
}


def main() -> None:
    src, out, run_dir, exp = sys.argv[1:5]
    flags = set(sys.argv[5:])
    steps = None
    if "--steps" in sys.argv:
        steps = sys.argv[sys.argv.index("--steps") + 1]
    arm = sys.argv[sys.argv.index("--arm") + 1] if "--arm" in sys.argv else None
    test_freq = sys.argv[sys.argv.index("--test-freq") + 1] if "--test-freq" in sys.argv else None
    val_max = sys.argv[sys.argv.index("--val-max") + 1] if "--val-max" in sys.argv else None
    save_freq = sys.argv[sys.argv.index("--save-freq") + 1] if "--save-freq" in sys.argv else None
    seed = None
    if "--seed" in sys.argv:
        seed = sys.argv[sys.argv.index("--seed") + 1]
    toks = shlex.split(open(src).read().strip())
    args = toks[toks.index("verl.trainer.main_ppo") + 1:]

    with initialize_config_dir(config_dir=f"{TREE}/verl/trainer/config", version_base=None):
        base = compose(config_name="ppo_trainer")
    exists = lambda k: OmegaConf.select(base, k, default="__MISSING__") != "__MISSING__"

    report = {"dropped": [], "mapped": [], "plus_added": [], "overridden": []}
    result: list[str] = []
    for a in args:
        k, _, v = a.partition("=")
        bare = k.lstrip("+")
        if bare.startswith(DROP_PREFIXES) or bare in DROP_KEYS:
            report["dropped"].append(bare)
            continue
        if bare in MAP_KEYS:
            nk, nv = MAP_KEYS[bare]
            report["mapped"].append(f"{bare}={v}  ->  {nk}={nv}")
            result.append(f"{nk}={nv}")
            continue
        if not k.startswith("+") and not exists(bare):
            report["plus_added"].append(bare)
            k = "+" + bare
        result.append(f"{k}={v}")

    if arm == "R":
        if "pureem" not in src.lower() and "W_" not in src:
            raise SystemExit("--arm R must start from the W (pure EM) command")
        for k, v in {
            "algorithm.planner_aux_control.mode": "pointwise_rloo_matched",
            "algorithm.planner_aux_control.target_psc_loss_weight": "0.02",
            "algorithm.planner_aux_control.beta": "1.0",
            "algorithm.planner_aux_control.epsilon": "1e-12",
            "algorithm.planner_aux_control.max_scale": "null",
            "algorithm.planner_aux_control.record_planner_base_grad": "true",
        }.items():
            result.append(f"+{k}={v}")
            report["plus_added"].append(k)
    if arm == "NRPSC":
        if "P_" not in src:
            raise SystemExit("--arm NRPSC must start from the P command (it overrides the PSC keys)")
        nr_weight = sys.argv[sys.argv.index("--nrpsc-weight") + 1] if "--nrpsc-weight" in sys.argv else "0.02"
        # Tie-break weight is OFF (0.0) by default: identical behaviour to before this flag existed. It is a
        # separate scalar from --nrpsc-weight, applied only to class pairs where the primary EM comparison ties
        # (most of all, the all-fail questions where EM gives the main objective no Planner signal at all); see
        # verl/trainer/ppo/nrpsc_objective.py: construct_node_pairs.
        tiebreak_weight = sys.argv[sys.argv.index("--nrpsc-tiebreak-weight") + 1] if "--nrpsc-tiebreak-weight" in sys.argv else "0.0"
        batch_q = sys.argv[sys.argv.index("--batch") + 1] if "--batch" in sys.argv else None
        nr = {
            "algorithm.planner_outcome_contrast.mode": "nrpsc",
            "algorithm.planner_outcome_contrast.minibatch_mode": "question_preserving",
            "algorithm.planner_outcome_contrast.loss_weight": nr_weight,
            "algorithm.planner_outcome_contrast.use_reward_eligibility": "true",
            "algorithm.planner_outcome_contrast.nrpsc_expected_siblings": "8",
            "algorithm.planner_outcome_contrast.nrpsc_idf_path": NRPSC_IDF_PATH,
            "algorithm.planner_outcome_contrast.nrpsc_idf_sha256": NRPSC_IDF_SHA256,
            "algorithm.planner_outcome_contrast.nrpsc_similarity_threshold": "0.8",
            "algorithm.planner_outcome_contrast.nrpsc_replay_budget": "0",
            "algorithm.planner_outcome_contrast.nrpsc_tiebreak_weight": tiebreak_weight,
            "qa_manager.workflow_rollout_n": "8",
        }
        if batch_q:
            nr["data.train_batch_size"] = batch_q
        nrpsc_over = nr
    elif arm == "W8":  # Workflow-RLOO baseline with the same N=8 siblings per question as NR-PSC
        if "W_" not in src:
            raise SystemExit("--arm W8 must start from the W command")
        nrpsc_over = {"qa_manager.workflow_rollout_n": "8"}
        if "--batch" in sys.argv:
            nrpsc_over["data.train_batch_size"] = sys.argv[sys.argv.index("--batch") + 1]
    else:
        nrpsc_over = {}
    ov = {
        "data.custom_cls.path": f"{TREE}/sippo/rl_dataset.py",
        "data.custom_cls.name": "SippoRLHFDataset",
        # JADE sends one row per call; the 0.9.1 AgentLoopManager chunks a batch evenly over its workers
        "actor_rollout_ref.rollout.agent.num_workers": "1",
        # the agent loop scores each step; the real reward comes from qa_manager, so use a placeholder scorer
        "reward.custom_reward_function.path": f"{TREE}/sippo/null_reward.py",
        "reward.custom_reward_function.name": "compute_score",
        "trainer.device": "npu",
        "trainer.experiment_name": exp,
        "trainer.default_local_dir": f"{run_dir}/ckpt",
        "trainer.test_freq": "-1",
    }
    ov.update(nrpsc_over)
    if steps:
        ov["trainer.total_training_steps"] = steps
    if test_freq is not None:
        ov["trainer.test_freq"] = test_freq
    ov["trainer.save_freq"] = save_freq if save_freq is not None else "-1"
    if val_max is not None:
        ov["data.val_max_samples"] = val_max
    if seed is not None:  # one knob -> data order, vLLM sampling, process RNG (see sippo/seed.py)
        ov["trainer.seed"] = seed
    if "--memory-safe" in flags:
        ov.update(MEMORY_SAFE)
    if "--fast" in flags:
        ov.update(FAST)
    if "--gpus" in sys.argv:  # data parallel: one full model replica per card (fsdp_size=1), see sippo/worker.py
        gpus = int(sys.argv[sys.argv.index("--gpus") + 1])
        ov["trainer.n_gpus_per_node"] = str(gpus)
        ov["actor_rollout_ref.actor.fsdp_config.fsdp_size"] = "1"
    for i, tok in enumerate(sys.argv):  # --set key=value (repeatable): final explicit overrides
        if tok == "--set":
            k, _, v = sys.argv[i + 1].partition("=")
            ov[k] = v
    keys = {r.partition("=")[0].lstrip("+"): i for i, r in enumerate(result)}
    for k, v in ov.items():
        if k in keys:
            pre = result[keys[k]].partition("=")[0]
            report["overridden"].append(f"{k}: {result[keys[k]].partition('=')[2][:40]} -> {v}")
            result[keys[k]] = f"{pre}={v}"
        else:
            plus = "" if exists(k) else "+"
            report["plus_added"].append(k)
            result.append(f"{plus}{k}={v}")

    # validate by composing the full config (strict) -- catches typos/unknown keys before any NPU is used
    ok = True
    try:
        with initialize_config_dir(config_dir=f"{TREE}/verl/trainer/config", version_base=None):
            cfg = compose(config_name="ppo_trainer", overrides=result)
        print("COMPOSE OK:", len(result), "overrides")
    except Exception as e:  # noqa: BLE001
        ok = False
        print("COMPOSE FAIL:", type(e).__name__, str(e).splitlines()[0][:300])
    open(out, "w").write("args=(\n" + "\n".join("  " + shlex.quote(a) for a in result) + "\n)\n")
    for k, v in report.items():
        print(f"[{k}] {len(v)}")
        for x in v:
            print("   ", x)
    sys.exit(0 if ok else 2)


if __name__ == "__main__":
    main()
