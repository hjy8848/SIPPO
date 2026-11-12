# SIPPO on Ascend NPU (verl 0.9.1 base)

This branch holds the NPU-native port of SIPPO/JADE: verl 0.9.1 plus the SIPPO workflow-RLOO / PSC / NR-PSC algorithm files.
It is a separate tree from `main` (the GPU snapshot); entry point is `python -m sippo.main <hydra overrides>`.

## Layout
- `sippo/` - trainer, actor update, workflow rollout, worker, launcher helpers (`build_args.py`)
- `verl/` - verl 0.9.1 with the SIPPO additions (`verl/trainer/ppo/nrpsc_objective.py`, `verl/workers/actor_sippo/`)
- `qa_manager/` - multi-agent DAG rollout and retrieval client
- `tests/` - CPU tests (NR-PSC objective, frozen-batch gradient comparison, 2-rank data-parallel equivalence)
- `launch_nrpsc.sh` - launcher (generates args from a historical JADE command, then starts `sippo.main`)

## What is in this snapshot
- NR-PSC objective (node-level, frozen TF-IDF, class-level Planner pairs), mode `nrpsc`.
- Data-parallel updates: one full model replica per NPU (`fsdp_size=1`), backward under `no_sync`, gradients averaged once per
  mini-batch. Verified on CPU (gloo, 2 ranks equal to the single-process update); 8-card NPU runs are in progress.
- Frozen-batch gradient audit (`sippo/actor_update.py: compare_gradients`, `sippo/nrpsc_rloo_audit.py`).
- Planner plans with more than 8 nodes are excluded from NR-PSC pairing and counted (`aux/nrpsc/oversize_plans`).

## Status / caveats
- Experimental; results are not final. Do not cite numbers from this branch without the run logs.
- NPU runs use a memory-safe profile for 61 GB cards (optimizer offload, gradient checkpointing, micro-batch 1).
  Not bitwise reproducible (async vLLM).
- `launch_nrpsc.sh`, `sippo/build_args.py` and the audit scripts still contain lab-specific absolute paths and service
  addresses (conda env, historical command files, IDF file, retriever URL). Adapt them before running elsewhere.
