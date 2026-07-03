# /// script
# requires-python = ">=3.10"
# dependencies = ["parakeet-mlx"]
# ///
# nvidia/parakeet-tdt-0.6b-v3 (MLX port) on the Apple Silicon GPU.
# Multilingual (25 EU languages incl. Latvian) with built-in language handling.
import argparse
import json
import time
from pathlib import Path

from parakeet_mlx import from_pretrained

p = argparse.ArgumentParser()
p.add_argument("--audio", required=True)
p.add_argument("--out", required=True)
p.add_argument("--repo", default="mlx-community/parakeet-tdt-0.6b-v3")
args = p.parse_args()

model = from_pretrained(args.repo)
t0 = time.time()
result = model.transcribe(args.audio, chunk_duration=120.0, overlap_duration=15.0)
elapsed = time.time() - t0

out = Path(args.out)
segs = [
    {"start": round(s.start, 2), "end": round(s.end, 2), "text": s.text.strip()}
    for s in getattr(result, "sentences", [])
]
(out / "transcript.txt").write_text(result.text.strip() + "\n")
if segs:
    (out / "segments.json").write_text(json.dumps(segs, ensure_ascii=False, indent=1))
(out / "meta.json").write_text(json.dumps({
    "backend": "parakeet-mlx",
    "model": args.repo,
    "device": "metal",
    "transcribe_seconds": round(elapsed, 1),
}, indent=1))
print(f"done in {elapsed:.0f}s")
