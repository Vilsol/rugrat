# ASR eval harness

Transcribes one audio file with a range of self-hosted models and reports
timing per model, so transcript quality (esp. Latvian with English words
sprinkled in) can be compared side by side.

## Prerequisites

- `uv` (https://docs.astral.sh/uv/) and `ffmpeg` on PATH
- macOS: Apple Silicon (runners use MLX on the Metal GPU)
- Linux: NVIDIA GPU + driver with CUDA >= 12.8 (Blackwell/sm_120 OK)

No Docker: containers on macOS cannot access the GPU. Each runner gets an
ephemeral env via `uv run` instead — created on first use, nothing persists
outside `cache/` and uv's own cache.

## Usage

```sh
./run.sh recording.mp3                 # all models supported on this platform
./run.sh recording.mp3 ailab-lv        # just one
./run.sh recording.mp3 large-v3 parakeet
LANGUAGE=lv ./run.sh recording.mp3     # force Latvian on stock whisper models
./run.sh clean                         # delete downloaded models + work files
```

## Models

| name           | model                                            | mac backend  | linux backend  |
|----------------|--------------------------------------------------|--------------|----------------|
| large-v3       | openai whisper-large-v3 (auto language detect)   | mlx-whisper  | faster-whisper |
| large-v3-turbo | whisper-large-v3-turbo (auto language detect)    | mlx-whisper  | faster-whisper |
| ailab-lv       | AiLab-IMCS-UL whisper-large-v3-lv-late-cv19      | mlx-whisper  | faster-whisper |
| ailab-cv17     | AiLab-IMCS-UL whisper-large-v3-lv-late-cv17      | transformers | transformers   |
| ailab-phono    | AiLab-IMCS-UL whisper-large-v3-lv-phono          | transformers | transformers   |
| parakeet       | nvidia parakeet-tdt-0.6b-v3 (25 EU langs)        | parakeet-mlx | NeMo           |
| gemma-e2b      | google gemma-4-E2B-it (audio-LLM, 28s chunks)    | transformers | transformers   |
| gemma-e4b      | google gemma-4-E4B-it (audio-LLM, 28s chunks)    | transformers | transformers   |
| gemma-12b      | google gemma-4-12B-it (audio-LLM, 28s chunks)    | — (skipped)  | transformers   |
| omnilingual-7b | meta omniASR_LLM_Unlimited_7B_v2                 | — (skipped)  | omnilingual-asr|
| canary         | nvidia canary-1b-v2 (25 EU langs, Latvian)       | — (skipped)  | NeMo           |

## Output

```
results/<audio-name>/<model>/
  transcript.txt   one line per segment
  segments.json    timestamps (whisper + parakeet-mlx)
  meta.json        backend, device, detected language, transcribe time
```

Model downloads land in `cache/hf` (largest: omnilingual 7B ~15GB;
whisper large-v3 ~3GB; parakeet ~0.6GB).

## What to look at

- WER feel: read transcripts against a stretch of audio you know.
- English insertions: grep the transcripts for the English words you expect;
  the failure mode is phonetic latvianization ("fīčers" for "features").
- `meta.json` `detected_language` shows what stock whisper auto-detected.
