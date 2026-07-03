# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = [
#   "nemo_toolkit[asr]>=2.4",
#   "torch>=2.7",
#   "cuda-python>=12.3",
# ]
# [tool.uv.sources]
# torch = { index = "pytorch-cu128" }
# [[tool.uv.index]]
# name = "pytorch-cu128"
# url = "https://download.pytorch.org/whl/cu128"
# explicit = true
# ///
# nvidia/canary-1b-v2 via NeMo on CUDA (no MLX build). Multilingual AED model;
# Latvian is listed as supported but has no published WER — this is a comparison
# point against the AiLab Latvian fine-tunes. NeMo chunks long audio internally.
import argparse
import json
import time
from pathlib import Path

import nemo.collections.asr as nemo_asr

p = argparse.ArgumentParser()
p.add_argument("--audio", required=True)
p.add_argument("--out", required=True)
p.add_argument("--model", default="nvidia/canary-1b-v2")
p.add_argument("--language", default="lv")
args = p.parse_args()

model = nemo_asr.models.ASRModel.from_pretrained(args.model)

t0 = time.time()
hyps = model.transcribe(
    [args.audio],
    source_lang=args.language,
    target_lang=args.language,
    timestamps=True,
)
elapsed = time.time() - t0

h = hyps[0]
text = h.text if hasattr(h, "text") else str(h)
out = Path(args.out)

segs = []
ts = getattr(h, "timestamp", None)
if isinstance(ts, dict) and ts.get("segment"):
    for s in ts["segment"]:
        segs.append({
            "start": round(float(s["start"]), 2),
            "end": round(float(s["end"]), 2),
            "text": (s.get("segment") or s.get("text", "")).strip(),
        })

(out / "transcript.txt").write_text(
    ("\n".join(s["text"] for s in segs) if segs else text.strip()) + "\n"
)
if segs:
    (out / "segments.json").write_text(json.dumps(segs, ensure_ascii=False, indent=1))
(out / "meta.json").write_text(json.dumps({
    "backend": "nemo",
    "model": args.model,
    "device": "cuda",
    "language_arg": args.language,
    "transcribe_seconds": round(elapsed, 1),
}, indent=1))
print(f"done in {elapsed:.0f}s")
