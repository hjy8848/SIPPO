#!/usr/bin/env bash
set -u

ROOT="${1:?usage: $0 OUTPUT_ROOT LOG_PATH}"
LOG="${2:?usage: $0 OUTPUT_ROOT LOG_PATH}"
ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

mkdir -p "$ROOT"
exec >>"$LOG" 2>&1
printf '[%s] queue_start host=%s pid=%s endpoint=%s\n' "$(date -Is)" "$(hostname)" "$$" "$ENDPOINT"

models=(
  'qwen25_3b_instruct|Qwen/Qwen2.5-3B-Instruct'
  'searchr1_qwen25_3b_it_ppo|PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-3b-it-em-ppo'
  'searchr1_qwen25_3b_it_grpo|PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-3b-it-em-grpo'
  'searchr1_qwen25_3b_base_ppo|PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-3b-em-ppo'
  'searchr1_qwen25_3b_base_grpo|PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-3b-em-grpo'
  'zerosearch_qwen25_3b_instruct|Alibaba-NLP/ZeroSearch_wiki_V2_Qwen2.5_3B_Instruct'
  'hiprag_qwen25_3b_grpo|qualidea1217/Qwen2.5-3B-Instruct-GRPO-HiPRAG'
  'hiprag_qwen25_3b_ppo|qualidea1217/Qwen2.5-3B-Instruct-PPO-HiPRAG'
  'qwen3_4b_instruct_2507|Qwen/Qwen3-4B-Instruct-2507'
  'infoseeker_4b|shreyess/infoseeker-4b'
  'orbit_4b_v01|orbit-ai/orbit-4b-v0.1'
  'reopd_4b|baohao/ReOPD_Search_Qwen3-4B-Instruct-2507_SFT-RL_to_Search_Qwen3-4B-Instruct-2507_SFT'
  'opd_4b|baohao/OPD_Search_Qwen3-4B-Instruct-2507_SFT-RL_to_Search_Qwen3-4B-Instruct-2507_SFT'
  'r1_searcher_7b|XXsongLALA/Qwen-2.5-7B-base-RAG-RL'
  's3_8b|pat-jj/s3-8-3-3-20steps'
  'deepresearcher_7b|GAIR/DeepResearcher-7b'
)

for item in "${models[@]}"; do
  name="${item%%|*}"
  repo="${item##*|}"
  target="$ROOT/$name"
  api="/tmp/baseline_${name}_$(id -u)_$$.json"
  printf '[%s] BEGIN name=%s repo=%s\n' "$(date -Is)" "$name" "$repo"

  if ! curl -fsSL --retry 12 --retry-delay 3 --retry-connrefused \
      --connect-timeout 15 --max-time 120 \
      "$ENDPOINT/api/models/$repo" -o "$api"; then
    printf '[%s] FAIL_API name=%s repo=%s\n' "$(date -Is)" "$name" "$repo"
    continue
  fi

  sha=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["sha"])' "$api" 2>/dev/null || true)
  if [ -z "$sha" ]; then
    printf '[%s] FAIL_META name=%s repo=%s\n' "$(date -Is)" "$name" "$repo"
    continue
  fi
  printf '[%s] REVISION name=%s sha=%s\n' "$(date -Is)" "$name" "$sha"

  mkdir -p "$target"
  python3 - "$api" >"$api.files" <<'PY'
import json
import sys

for item in json.load(open(sys.argv[1])).get("siblings", []):
    name = item.get("rfilename")
    if name:
        print(name)
PY

  failed=0
  while IFS= read -r rel; do
    out="$target/$rel"
    mkdir -p "$(dirname "$out")"
    url="$ENDPOINT/$repo/resolve/$sha/$rel"
    printf '[%s] FILE name=%s rel=%s\n' "$(date -Is)" "$name" "$rel"
    if ! curl -fL --retry 20 --retry-delay 3 --retry-connrefused \
        --connect-timeout 20 --max-time 0 -C - \
        -o "$out.part" "$url"; then
      printf '[%s] FAIL_FILE name=%s rel=%s\n' "$(date -Is)" "$name" "$rel"
      failed=1
      break
    fi
    mv -f "$out.part" "$out"
  done <"$api.files"

  if [ "$failed" -eq 0 ]; then
    printf '%s %s\n' "$sha" "$repo" >"$target/.revision"
    printf '[%s] DONE name=%s repo=%s sha=%s\n' "$(date -Is)" "$name" "$repo" "$sha"
  else
    printf '[%s] PARTIAL name=%s repo=%s sha=%s\n' "$(date -Is)" "$name" "$repo" "$sha"
  fi
done

printf '[%s] queue_end\n' "$(date -Is)"
