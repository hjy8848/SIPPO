#!/bin/bash
# NR-PSC launcher for the independent copy. usage: launch_nrpsc.sh TAG NPU SEED [STEPS] -- extra build_args flags via $EXTRA
# Everything resolves inside this tree; the original NPU tree is never referenced. A real run starts only when called.
set -e
TAG=$1; NPU=$2; SEED=${3:-42}; STEPS=${4:-1}
N=$(cd "$(dirname "$0")" && pwd)
E=/home/hejingyu/miniconda3_arm/envs/jade-sippo-npu-20261107
H=/home/hejingyu/projects/JADE-sippo-npu-20261107/_setup_logs/hist_cmds
R=$N/_runs/$TAG
mkdir -p $R
source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh >/dev/null 2>&1
cd /tmp
TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONPATH=$N $E/bin/python $N/sippo/build_args.py $H/${CMDFILE:-P_initial.cmd} $R/args.sh $R ${EXPPREFIX-NRPSC-}$TAG --memory-safe --steps $STEPS --seed $SEED --arm ${ARM:-NRPSC} $EXTRA > $R/build_args.log 2>&1 || { echo "build_args failed"; tail -5 $R/build_args.log; exit 2; }
(cd $N && find sippo verl/trainer/ppo verl/workers/config verl/experimental/agent_loop qa_manager -name '*.py' | sort | xargs md5sum) > $R/code_manifest.md5
cat > $R/run.sh <<RUN
#!/bin/bash
N=$N
E=$E
R=$R
source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh >/dev/null 2>&1
set +e; source /usr/local/Ascend/nnal/atb/set_env.sh >/dev/null 2>&1; set -e
source \$R/args.sh
export ASCEND_RT_VISIBLE_DEVICES=$NPU PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false HYDRA_FULL_ERROR=1
export PYTHONPATH=\${N}:\${PYTHONPATH} VLLM_USE_V1=1 VLLM_ASCEND_ENABLE_NZ=0
export SIPPO_TREE=\$N
export RAY_TMPDIR=/home/hejingyu/.rbray/nrpsc_$TAG; mkdir -p \$RAY_TMPDIR
${RUN_ENV}
cd \$N
exec \$E/bin/python -u -m sippo.main "\${args[@]}"
RUN
chmod +x $R/run.sh
if [ "$NO_START" = "1" ]; then echo "$TAG prepared (not started) NPU $NPU"; exit 0; fi
nohup setsid $R/run.sh > $R/run.log 2>&1 &
echo "$TAG PID $! NPU $NPU $(date -u +%FT%TZ)"