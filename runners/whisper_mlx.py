# /// script
# requires-python = ">=3.10"
# dependencies = ["mlx-whisper"]
# ///
# Runs on the Apple Silicon GPU via MLX/Metal.
import argparse
import json
import time
from pathlib import Path

import mlx_whisper
from huggingface_hub import snapshot_download

p = argparse.ArgumentParser()
p.add_argument("--audio", required=True)
p.add_argument("--out", required=True)
p.add_argument("--repo", required=True)
p.add_argument("--language", default="auto")
args = p.parse_args()

# mlx_whisper.load_model wants weights.safetensors/weights.npz; some mlx-community
# repos (e.g. whisper-large-v3-lv-late-cv19) ship model.safetensors instead. Resolve
# the snapshot and alias it so the loader finds it. No-op for repos that ship npz.
repo = Path(snapshot_download(args.repo))
if not (repo / "weights.safetensors").exists() and not (repo / "weights.npz").exists():
    src = repo / "model.safetensors"
    if src.exists():
        (repo / "weights.safetensors").symlink_to(src.name)

lang = None if args.language == "auto" else args.language
t0 = time.time()
result = mlx_whisper.transcribe(args.audio, path_or_hf_repo=str(repo), language=lang, verbose=None)
elapsed = time.time() - t0

out = Path(args.out)
segs = [
    {"start": round(s["start"], 2), "end": round(s["end"], 2), "text": s["text"].strip()}
    for s in result.get("segments", [])
]
(out / "transcript.txt").write_text("\n".join(s["text"] for s in segs) + "\n")
(out / "segments.json").write_text(json.dumps(segs, ensure_ascii=False, indent=1))
(out / "meta.json").write_text(json.dumps({
    "backend": "mlx-whisper",
    "model": args.repo,
    "device": "metal",
    "language_arg": args.language,
    "detected_language": result.get("language"),
    "transcribe_seconds": round(elapsed, 1),
}, indent=1))
print(f"done in {elapsed:.0f}s, language: {result.get('language')}")
