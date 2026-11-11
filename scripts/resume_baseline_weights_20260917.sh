#!/usr/bin/env bash
set -u

ROOT="${1:?usage: $0 OUTPUT_ROOT LOG_PATH}"
LOG="${2:?usage: $0 OUTPUT_ROOT LOG_PATH}"
ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

mkdir -p "$ROOT"
exec >>"$LOG" 2>&1
printf '[%s] resume_start host=%s pid=%s endpoint=%s\n' "$(date -Is)" "$(hostname)" "$$" "$ENDPOINT"

models=(
  'hiprag_qwen25_3b_grpo|qualidea1217/Qwen2.5-3B-Instruct-GRPO-HiPRAG'
  'hiprag_qwen25_3b_ppo|qualidea1217/Qwen2.5-3B-Instruct-PPO-HiPRAG'
  'infoseeker_4b|shreyess/infoseeker-4b'
  'reopd_4b|baohao/ReOPD_Search_Qwen3-4B-Instruct-2507_SFT-RL_to_Search_Qwen3-4B-Instruct-2507_SFT'
  'opd_4b|baohao/OPD_Search_Qwen3-4B-Instruct-2507_SFT-RL_to_Search_Qwen3-4B-Instruct-2507_SFT'
  'orbit_4b_v01|orbit-ai/orbit-4b-v0.1'
)

for item in "${models[@]}"; do
  name="${item%%|*}"
  repo="${item##*|}"
  target="$ROOT/$name"
  api="/tmp/baseline_resume_${name}_$(id -u)_$$.json"
  printf '[%s] BEGIN name=%s repo=%s\n' "$(date -Is)" "$name" "$repo"

  if ! curl -fsSL --retry 12 --retry-delay 3 --retry-connrefused \
      --connect-timeout 15 --max-time 120 \
      "$ENDPOINT/api/models/$repo" -o "$api"; then
    printf '[%s] FAIL_API name=%s repo=%s\n' "$(date -Is)" "$name" "$repo"
    continue
  fi
  sha=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["sha"])' "$api" 2>/dev/null || true)
  [ -n "$sha" ] || { printf '[%s] FAIL_META name=%s\n' "$(date -Is)" "$name"; continue; }
  printf '[%s] REVISION name=%s sha=%s\n' "$(date -Is)" "$name" "$sha"

  mkdir -p "$target"
  python3 - "$api" >"$api.files" <<'PY'
import json
import sys
for item in json.load(open(sys.argv[1])).get("siblings", []):
    name = item.get("rfilename")
    if name:
        print(f"{name}\t{item.get('size', '')}")
PY

  failed=0
  while IFS=$'\t' read -r rel expected_size; do
    [ -n "$rel" ] || continue
    out="$target/$rel"
    part="$out.part"
    mkdir -p "$(dirname "$out")"
    if [ -f "$out" ] && [ ! -f "$part" ]; then
      printf '[%s] SKIP_EXISTING name=%s rel=%s\n' "$(date -Is)" "$name" "$rel"
      continue
    fi
    if [ -f "$out" ] && [ -n "$expected_size" ] && [ "$(stat -c%s "$out")" = "$expected_size" ]; then
      printf '[%s] SKIP_COMPLETE name=%s rel=%s\n' "$(date -Is)" "$name" "$rel"
      continue
    fi
    if [ -f "$part" ] && [ -n "$expected_size" ] && [ "$(stat -c%s "$part")" = "$expected_size" ]; then
      mv -f "$part" "$out"
      printf '[%s] PROMOTE_COMPLETE name=%s rel=%s\n' "$(date -Is)" "$name" "$rel"
      continue
    fi
    url="$ENDPOINT/$repo/resolve/$sha/$rel"
    printf '[%s] FILE name=%s rel=%s current=%s expected=%s\n' "$(date -Is)" "$name" "$rel" "$(stat -c%s "$part" 2>/dev/null || echo 0)" "$expected_size"
    if ! curl -fL --retry 20 --retry-delay 3 --retry-connrefused \
        --speed-time 60 --speed-limit 1024 --connect-timeout 20 --max-time 0 \
        -C - -o "$part" "$url"; then
      printf '[%s] FAIL_FILE name=%s rel=%s\n' "$(date -Is)" "$name" "$rel"
      failed=1
      break
    fi
    if [ -n "$expected_size" ] && [ "$(stat -c%s "$part")" != "$expected_size" ]; then
      printf '[%s] FAIL_SIZE name=%s rel=%s actual=%s expected=%s\n' "$(date -Is)" "$name" "$rel" "$(stat -c%s "$part")" "$expected_size"
      failed=1
      break
    fi
    mv -f "$part" "$out"
  done <"$api.files"

  if [ "$failed" -eq 0 ]; then
    printf '%s %s\n' "$sha" "$repo" >"$target/.revision"
    printf '[%s] DONE name=%s repo=%s sha=%s\n' "$(date -Is)" "$name" "$repo" "$sha"
  else
    printf '[%s] PARTIAL name=%s repo=%s sha=%s\n' "$(date -Is)" "$name" "$repo" "$sha"
  fi
done
printf '[%s] resume_end\n' "$(date -Is)"
