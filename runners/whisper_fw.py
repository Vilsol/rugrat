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
        d = os.path.dirname(pkg.__file__)
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
p.add_argument("--repo", required=True)
p.add_argument("--subfolder", default=None)
p.add_argument("--language", default="auto")
args = p.parse_args()

model_path = args.repo
if args.subfolder:
    from huggingface_hub import snapshot_download

    root = snapshot_download(args.repo, allow_patterns=[f"{args.subfolder}/*"])
    model_path = str(Path(root) / args.subfolder)

model = WhisperModel(model_path, device="cuda", compute_type="float16")

lang = None if args.language == "auto" else args.language
t0 = time.time()
segments, info = model.transcribe(args.audio, language=lang)

out = Path(args.out)
segs = []
with (out / "transcript.txt").open("w") as f:
    for s in segments:
        f.write(s.text.strip() + "\n")
        segs.append({"start": round(s.start, 2), "end": round(s.end, 2), "text": s.text.strip()})
elapsed = time.time() - t0

(out / "segments.json").write_text(json.dumps(segs, ensure_ascii=False, indent=1))
(out / "meta.json").write_text(json.dumps({
    "backend": "faster-whisper",
    "model": args.repo + (f"/{args.subfolder}" if args.subfolder else ""),
    "device": "cuda/float16",
    "language_arg": args.language,
    "detected_language": info.language,
    "language_probability": round(info.language_probability, 3),
    "transcribe_seconds": round(elapsed, 1),
}, indent=1))
print(f"done in {elapsed:.0f}s, detected language: {info.language} (p={info.language_probability:.2f})")
