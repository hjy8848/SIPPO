set -x

hotpot_qa_train_path=/mnt/tidal-alsh01/usr/datasets/data/verl_format_data/hotpotqa/train_verl.parquet
hotpot_qa_test_path=/mnt/tidal-alsh01/usr/datasets/data/verl_format_data/hotpotqa/test_verl.parquet

musique_test_path=/mnt/tidal-alsh01/usr/datasets/data/verl_format_data/musique/test_verl.parquet
wikimultihopqa_test_path=/mnt/tidal-alsh01/usr/datasets/data/verl_format_data/2wikimultihopqa/test_verl.parquet
bamboogle_test_path=/mnt/tidal-alsh01/usr/datasets/data/verl_format_data/bamboogle/test_verl.parquet

nq_search_test_path=/mnt/tidal-alsh01/usr/datasets/data/verl_format_data/nq_search/test_verl.parquet
ambig_qa_test_path=/mnt/tidal-alsh01/usr/datasets/data/verl_format_data/ambig_qa/test_verl.parquet
popqa_test_path=/mnt/tidal-alsh01/usr/datasets/data/verl_format_data/popqa/test_verl.parquet

train_files="['$hotpot_qa_train_path']"
test_files="['$hotpot_qa_test_path']"

# WandB 登录
export WANDB_API_KEY="${WANDB_API_KEY:?Set WANDB_API_KEY}"
# vllm
export VLLM_USE_V1=1

echo ' ***************** export vllm v1 ***************** '
echo ' ***************** export wandb api_key ***************** '

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=gae \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=64 \
    data.val_batch_size=1024 \
    data.max_prompt_length=3072 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=/mnt/tidal-alsh01/usr/research_project/adaptive_joint_optim/sft/sft_ckpt/Qwen2.5-7B-Instruct/v2 \
    actor_rollout_ref.model.enable_gradient_checkpointing=False \
    actor_rollout_ref.actor.optim.lr=5e-7 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    critic.optim.lr=1e-5 \
    critic.model.use_remove_padding=True \
    critic.model.path=/mnt/tidal-alsh01/usr/research_project/adaptive_joint_optim/sft/sft_ckpt/Qwen2.5-7B-Instruct/v2 \
    critic.model.enable_gradient_checkpointing=False \
    critic.ppo_micro_batch_size_per_gpu=2 \
    critic.model.fsdp_config.param_offload=False \
    critic.model.fsdp_config.optimizer_offload=False \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["wandb"]' \
    trainer.project_name='adaptive_joint_optim_nq_single-hop' \
    trainer.experiment_name='single-hop_M-ASK_bs128_lr5e-7_wosft_5doc_B' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=10000000 \
    trainer.test_freq=10000000 \
    trainer.total_epochs=10 $@ \
    data.return_raw_chat=True \
    actor_rollout_ref.rollout.mode=async \
    trainer.val_before_train=True \

# actor_rollout_ref.rollout.temperature=0.7 \
# actor_rollout_ref.rollout.top_p=0.8 \

# actor_rollout_ref.actor.use_kl_loss=True \
# actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
