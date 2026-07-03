# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "transformers>=4.44",
#   "torch>=2.4",
#   "accelerate>=0.33",
#   "soundfile>=0.12",
#   "numpy",
# ]
# ///
# Any raw Hugging Face Whisper fine-tune, no MLX/CT2 conversion needed, on
# Apple Silicon (MPS) or CUDA. Used for the AiLab-IMCS-UL Latvian fine-tunes
# that ship only standard HF weights. Native sequential long-form decoding
# (same algorithm as whisper_mlx / faster-whisper) for accuracy over speed;
# fp32 on MPS. Blackwell/sm_120: if torch lacks sm_120, install the cu128 build.
import argparse
import json
import time
from pathlib import Path

import soundfile as sf
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

p = argparse.ArgumentParser()
p.add_argument("--audio", required=True)
p.add_argument("--out", required=True)
p.add_argument("--repo", required=True)
p.add_argument("--language", default="auto")
args = p.parse_args()

if torch.cuda.is_available():
    device, dtype = "cuda", torch.float16
elif torch.backends.mps.is_available():
    device, dtype = "mps", torch.float32  # fp16 on MPS is unstable
else:
    device, dtype = "cpu", torch.float32

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    args.repo, torch_dtype=dtype, low_cpu_mem_usage=True
)
model.to(device)
model.eval()
processor = AutoProcessor.from_pretrained(args.repo)

audio, sr = sf.read(args.audio, dtype="float32")
if audio.ndim > 1:
    audio = audio.mean(axis=1)

# truncation=False keeps the full audio so generate() uses Whisper's native
# sequential long-form decoding (30s windows, condition-on-previous) rather than
# the less-accurate chunked pipeline path.
inputs = processor(
    audio,
    sampling_rate=sr,
    return_tensors="pt",
    truncation=False,
    padding="longest",
    return_attention_mask=True,
)
inputs = inputs.to(device)
inputs["input_features"] = inputs["input_features"].to(dtype)

gen = dict(
    task="transcribe",
    return_timestamps=True,
    return_segments=True,
    condition_on_prev_tokens=True,
    # OpenAI-default temperature fallback + thresholds: recover from repetition
    # loops / low-confidence windows without giving up greedy accuracy.
    temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
    compression_ratio_threshold=2.4,
    logprob_threshold=-1.0,
    no_speech_threshold=0.6,
)
if args.language != "auto":
    gen["language"] = args.language

t0 = time.time()
with torch.no_grad():
    outputs = model.generate(**inputs, **gen)
elapsed = time.time() - t0

out = Path(args.out)
segs = []
for seg in outputs["segments"][0]:
    text = processor.decode(seg["tokens"], skip_special_tokens=True).strip()
    segs.append({
        "start": round(float(seg["start"]), 2),
        "end": round(float(seg["end"]), 2),
        "text": text,
    })

(out / "transcript.txt").write_text("\n".join(s["text"] for s in segs) + "\n")
(out / "segments.json").write_text(json.dumps(segs, ensure_ascii=False, indent=1))
(out / "meta.json").write_text(json.dumps({
    "backend": "transformers",
    "model": args.repo,
    "device": f"{device}/{str(dtype).split('.')[-1]}",
    "language_arg": args.language,
    "decode": "sequential-longform",
    "transcribe_seconds": round(elapsed, 1),
}, indent=1))
print(f"done in {elapsed:.0f}s, language: {args.language}")
