#!/usr/bin/env bash
# ASR eval harness: stand up model -> transcribe -> tear down, per model.
# Usage: ./run.sh <audio-file> [model ...]
#        ./run.sh clean
# Models: large-v3 large-v3-turbo ailab-lv parakeet omnilingual-7b
# macOS (Apple Silicon): MLX/Metal backends. Linux + NVIDIA: CUDA backends.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
export HF_HOME="$ROOT/cache/hf"
LANGUAGE="${LANGUAGE:-auto}"   # LANGUAGE=lv ./run.sh ... to force Latvian on stock whisper

# faster-whisper decoding knobs (apply to large-v3, large-v3-turbo, ailab-lv):
VAD="${VAD:-1}"                 # Silero VAD filter: strips silence/applause (kills hallucination)
BEAM="${BEAM:-8}"              # beam size (default whisper is 5)
NO_REPEAT="${NO_REPEAT:-3}"    # no_repeat_ngram_size: 3 kills verbatim loops
REP_PENALTY="${REP_PENALTY:-1.0}"  # 1.0=off; >1 corrupts legit Latvian repeats (būs..būs, laicīgi..laicīgi)
INITIAL_PROMPT="${INITIAL_PROMPT:-}"  # INITIAL_PROMPT="terms: features roadmap KPI ..." to bias spelling
HOTWORDS="${HOTWORDS:-}"              # HOTWORDS="features roadmap KPI" (used when INITIAL_PROMPT unset)

if [[ "${1:-}" == "clean" ]]; then
  rm -rf "$ROOT/cache" "$ROOT/work"
  echo "removed cache/ and work/ (results/ kept)"
  exit 0
fi

AUDIO="${1:?usage: ./run.sh <audio-file> [model ...]}"
shift || true
command -v uv >/dev/null || { echo "uv is required: https://docs.astral.sh/uv/"; exit 1; }
command -v ffmpeg >/dev/null || { echo "ffmpeg is required"; exit 1; }

OS="$(uname -s)"
if [[ "$OS" == "Darwin" ]]; then
  PLATFORM=mac
elif command -v nvidia-smi >/dev/null 2>&1; then
  PLATFORM=cuda
else
  echo "Linux without NVIDIA GPU detected — this would be CPU-only and take hours. Aborting."
  exit 1
fi

DEFAULT_MODELS=(large-v3 large-v3-turbo ailab-lv ailab-cv17 ailab-phono parakeet gemma-e2b gemma-e4b)
[[ "$PLATFORM" == "cuda" ]] && DEFAULT_MODELS+=(omnilingual-7b canary gemma-12b)
MODELS=("${@:-${DEFAULT_MODELS[@]}}")

# common faster-whisper tuning flags, assembled from the env knobs above
FW_TUNE=(--vad "$VAD" --beam-size "$BEAM" --no-repeat-ngram "$NO_REPEAT" --repetition-penalty "$REP_PENALTY")
[[ -n "$INITIAL_PROMPT" ]] && FW_TUNE+=(--initial-prompt "$INITIAL_PROMPT")
[[ -n "$HOTWORDS" ]] && FW_TUNE+=(--hotwords "$HOTWORDS")

# --- preprocess: 16kHz mono wav, done once per input file ---
mkdir -p "$ROOT/work" "$ROOT/results"
BASE="$(basename "${AUDIO%.*}")"
WAV="$ROOT/work/$BASE.16k.wav"
if [[ ! -f "$WAV" || "$AUDIO" -nt "$WAV" ]]; then
  echo "==> converting to 16kHz mono wav"
  ffmpeg -y -loglevel error -i "$AUDIO" -ac 1 -ar 16000 -c:a pcm_s16le "$WAV"
fi
DUR="$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$WAV")"
printf '==> audio: %s (%.0f s)\n' "$BASE" "$DUR"

declare -a SUMMARY

run_model() {
  local name="$1" runner="$2"; shift 2
  local outdir="$ROOT/results/$BASE/$name"
  mkdir -p "$outdir"
  echo
  echo "=== $name ($runner) ==="
  local start end rc=0
  start=$(date +%s)
  uv run "$ROOT/runners/$runner" --audio "$WAV" --out "$outdir" "$@" || rc=$?
  end=$(date +%s)
  local wall=$((end - start))
  if [[ $rc -eq 0 ]]; then
    local rtf
    rtf=$(awk -v w="$wall" -v d="$DUR" 'BEGIN{printf "%.2f", d/w}')
    SUMMARY+=("$(printf '%-16s %6ss  %5sx realtime' "$name" "$wall" "$rtf")")
  else
    SUMMARY+=("$(printf '%-16s FAILED (exit %s)' "$name" "$rc")")
  fi
}

for m in "${MODELS[@]}"; do
  case "$PLATFORM:$m" in
    mac:large-v3)        run_model "$m" whisper_mlx.py --repo mlx-community/whisper-large-v3-mlx --language "$LANGUAGE" ;;
    mac:large-v3-turbo)  run_model "$m" whisper_mlx.py --repo mlx-community/whisper-large-v3-turbo --language "$LANGUAGE" ;;
    mac:ailab-lv)        run_model "$m" whisper_mlx.py --repo mlx-community/whisper-large-v3-lv-late-cv19 --language lv ;;
    mac:ailab-cv17)      run_model "$m" whisper_hf.py --repo AiLab-IMCS-UL/whisper-large-v3-lv-late-cv17 --language lv ;;
    mac:ailab-phono)     run_model "$m" whisper_hf.py --repo AiLab-IMCS-UL/whisper-large-v3-lv-phono --language lv ;;
    mac:parakeet)        run_model "$m" parakeet_run.py ;;
    mac:gemma-e2b)       run_model "$m" gemma4_hf.py --repo google/gemma-4-E2B-it --language Latvian ;;
    mac:gemma-e4b)       run_model "$m" gemma4_hf.py --repo google/gemma-4-E4B-it --language Latvian ;;
    mac:omnilingual-7b)  echo "skipping $m: CUDA-only (7B on Mac is impractical)"; SUMMARY+=("$(printf '%-16s SKIPPED (cuda only)' "$m")") ;;
    mac:canary)          echo "skipping $m: CUDA-only (NeMo)"; SUMMARY+=("$(printf '%-16s SKIPPED (cuda only)' "$m")") ;;
    mac:gemma-12b)       echo "skipping $m: 12B impractical on Mac (use gemma-e2b/e4b)"; SUMMARY+=("$(printf '%-16s SKIPPED (cuda only)' "$m")") ;;
    cuda:large-v3)       run_model "$m" whisper_fw.py --repo Systran/faster-whisper-large-v3 --language "$LANGUAGE" "${FW_TUNE[@]}" ;;
    cuda:large-v3-turbo) run_model "$m" whisper_fw.py --repo deepdml/faster-whisper-large-v3-turbo-ct2 --language "$LANGUAGE" "${FW_TUNE[@]}" ;;
    cuda:ailab-lv)       CV19_CT2="$ROOT/cache/ct2/cv19-fp16"
                         uv run "$ROOT/runners/ct2_convert.py" --repo AiLab-IMCS-UL/whisper-large-v3-lv-late-cv19 --out "$CV19_CT2" --quant float16
                         run_model "$m" whisper_fw.py --model-path "$CV19_CT2" --language lv "${FW_TUNE[@]}" ;;
    cuda:ailab-cv17)     run_model "$m" whisper_hf.py --repo AiLab-IMCS-UL/whisper-large-v3-lv-late-cv17 --language lv ;;
    cuda:ailab-phono)    run_model "$m" whisper_hf.py --repo AiLab-IMCS-UL/whisper-large-v3-lv-phono --language lv ;;
    cuda:parakeet)       run_model "$m" parakeet_nemo.py ;;
    cuda:gemma-e2b)      run_model "$m" gemma4_hf.py --repo google/gemma-4-E2B-it --language Latvian ;;
    cuda:gemma-e4b)      run_model "$m" gemma4_hf.py --repo google/gemma-4-E4B-it --language Latvian ;;
    cuda:gemma-12b)      run_model "$m" gemma4_hf.py --repo google/gemma-4-12B-it --language Latvian ;;
    cuda:omnilingual-7b) run_model "$m" omnilingual.py ;;
    cuda:canary)         run_model "$m" canary_nemo.py --language lv ;;
    *) echo "unknown model: $m"; SUMMARY+=("$(printf '%-16s UNKNOWN' "$m")") ;;
  esac
done

echo
echo "=== summary ($PLATFORM, language=$LANGUAGE) ==="
printf '%s\n' "${SUMMARY[@]}"
echo "transcripts: results/$BASE/<model>/transcript.txt (+ segments.json where available)"
