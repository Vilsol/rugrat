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
# nvidia/parakeet-tdt-0.6b-v3 via NeMo on CUDA. torch cu128 pinned for
# Blackwell (sm_120). Local attention lets it take the full 1.5h file in one
# pass (full attention caps out around 24 min).
import argparse
import json
import time
from pathlib import Path

import nemo.collections.asr as nemo_asr

p = argparse.ArgumentParser()
p.add_argument("--audio", required=True)
p.add_argument("--out", required=True)
p.add_argument("--model", default="nvidia/parakeet-tdt-0.6b-v3")
args = p.parse_args()

model = nemo_asr.models.ASRModel.from_pretrained(args.model)
model.change_attention_model("rel_pos_local_attn", [256, 256])
model.change_subsampling_conv_chunking_factor(1)

t0 = time.time()
hyps = model.transcribe([args.audio])
elapsed = time.time() - t0

text = hyps[0].text if hasattr(hyps[0], "text") else str(hyps[0])
out = Path(args.out)
(out / "transcript.txt").write_text(text.strip() + "\n")
(out / "meta.json").write_text(json.dumps({
    "backend": "nemo",
    "model": args.model,
    "device": "cuda",
    "attention": "rel_pos_local_attn[256,256]",
    "transcribe_seconds": round(elapsed, 1),
}, indent=1))
print(f"done in {elapsed:.0f}s")
