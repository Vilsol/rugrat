# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "faster-whisper>=1.1.0",
#   "ctranslate2>=4.5.0",
#   "nvidia-cublas-cu12",
#   "nvidia-cudnn-cu12>=9",
# ]
# ///
# Blackwell (sm_120) note: needs ctranslate2>=4.5.0; int8 compute crashes on
# sm_120 in older builds, so we decode in float16 on GPU regardless of how the
# weights are stored.
import argparse
import ctypes
import json
import os
import time
from pathlib import Path


def _preload_cuda_libs():
    # CTranslate2 dlopen()s libcublas/libcudnn by soname, but a uv ephemeral env
    # has no system CUDA on the loader path. Preload the pip-provided libs with
    # RTLD_GLOBAL so the loader resolves them without setting LD_LIBRARY_PATH.
    try:
        import nvidia.cublas.lib as cublas_lib
        import nvidia.cudnn.lib as cudnn_lib
    except ImportError:
        return
    for pkg in (cublas_lib, cudnn_lib):  # cublas first: cudnn depends on it
        for d in list(pkg.__path__):  # namespace packages: __file__ is None
            for name in sorted(os.listdir(d)):
                if ".so" in name:
                    try:
                        ctypes.CDLL(os.path.join(d, name), mode=ctypes.RTLD_GLOBAL)
                    except OSError:
                        pass


_preload_cuda_libs()

from faster_whisper import WhisperModel

p = argparse.ArgumentParser()
p.add_argument("--audio", required=True)
p.add_argument("--out", required=True)
p.add_argument("--repo")
p.add_argument("--subfolder", default=None)
p.add_argument("--model-path", default=None, help="local CT2 dir (skips repo download)")
p.add_argument("--language", default="auto")
# --- decoding knobs (see README "turning knobs") ---
p.add_argument("--vad", type=int, default=1, help="1=Silero VAD filter, 0=off")
p.add_argument("--min-silence-ms", type=int, default=500)
p.add_argument("--beam-size", type=int, default=8)
p.add_argument("--no-repeat-ngram", type=int, default=3, help="0=off; 3 kills verbatim loops")
p.add_argument("--repetition-penalty", type=float, default=1.1)
p.add_argument("--initial-prompt", default=None, help="glossary/context to bias spelling")
p.add_argument("--hotwords", default=None, help="hint phrases (used when --initial-prompt unset)")
args = p.parse_args()

if args.model_path:
    model_path = args.model_path
elif args.subfolder:
    from huggingface_hub import snapshot_download

    root = snapshot_download(args.repo, allow_patterns=[f"{args.subfolder}/*"])
    model_path = str(Path(root) / args.subfolder)
else:
    model_path = args.repo

model = WhisperModel(model_path, device="cuda", compute_type="float16")

lang = None if args.language == "auto" else args.language
vad_params = dict(min_silence_duration_ms=args.min_silence_ms) if args.vad else None
t0 = time.time()
segments, info = model.transcribe(
    args.audio,
    language=lang,
    beam_size=args.beam_size,
    no_repeat_ngram_size=args.no_repeat_ngram,
    repetition_penalty=args.repetition_penalty,
    vad_filter=bool(args.vad),
    vad_parameters=vad_params,
    initial_prompt=args.initial_prompt or None,
    hotwords=args.hotwords or None,
)

out = Path(args.out)
segs = []
with (out / "transcript.txt").open("w") as f:
    for s in segments:
        f.write(s.text.strip() + "\n")
        segs.append({"start": round(s.start, 2), "end": round(s.end, 2), "text": s.text.strip()})
elapsed = time.time() - t0

model_id = args.model_path or (args.repo + (f"/{args.subfolder}" if args.subfolder else ""))
(out / "segments.json").write_text(json.dumps(segs, ensure_ascii=False, indent=1))
(out / "meta.json").write_text(json.dumps({
    "backend": "faster-whisper",
    "model": model_id,
    "device": "cuda/float16",
    "language_arg": args.language,
    "detected_language": info.language,
    "language_probability": round(info.language_probability, 3),
    "vad": bool(args.vad),
    "beam_size": args.beam_size,
    "no_repeat_ngram": args.no_repeat_ngram,
    "repetition_penalty": args.repetition_penalty,
    "initial_prompt": args.initial_prompt,
    "hotwords": args.hotwords,
    "transcribe_seconds": round(elapsed, 1),
}, indent=1))
print(f"done in {elapsed:.0f}s, detected language: {info.language} (p={info.language_probability:.2f})")
