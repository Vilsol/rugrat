# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = [
#   "omnilingual-asr",
#   "torch==2.7.1",
#   "torchaudio==2.7.1",
# ]
# [tool.uv.sources]
# torch = { index = "pytorch-cu128" }
# torchaudio = { index = "pytorch-cu128" }
# fairseq2 = { index = "fairseq2-cu128" }
# fairseq2n = { index = "fairseq2-cu128" }
# [[tool.uv.index]]
# name = "pytorch-cu128"
# url = "https://download.pytorch.org/whl/cu128"
# explicit = true
# [[tool.uv.index]]
# name = "fairseq2-cu128"
# url = "https://fair.pkg.atmeta.com/fairseq2/whl/pt2.7.1/cu128"
# explicit = true
# ///
# Meta Omnilingual ASR, "Unlimited" variant (handles arbitrary-length audio,
# no manual VAD splitting). CUDA/Linux only in this harness.
# Default runs unconditioned (model infers language); pass --lang lvs_Latn to force.
import argparse
import json
import time
from pathlib import Path

from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline

p = argparse.ArgumentParser()
p.add_argument("--audio", required=True)
p.add_argument("--out", required=True)
p.add_argument("--model-card", default="omniASR_LLM_Unlimited_7B_v2")
p.add_argument("--lang", default="auto")
args = p.parse_args()

pipeline = ASRInferencePipeline(model_card=args.model_card)

kwargs = {} if args.lang == "auto" else {"lang": [args.lang]}
t0 = time.time()
try:
    results = pipeline.transcribe([args.audio], batch_size=1, **kwargs)
except TypeError:
    # some pipeline versions require lang conditioning
    results = pipeline.transcribe([args.audio], batch_size=1, lang=["lvs_Latn"])
    args.lang = "lvs_Latn (forced: pipeline requires lang)"
elapsed = time.time() - t0

text = results[0] if isinstance(results[0], str) else getattr(results[0], "text", str(results[0]))
out = Path(args.out)
(out / "transcript.txt").write_text(text.strip() + "\n")
(out / "meta.json").write_text(json.dumps({
    "backend": "omnilingual-asr",
    "model": args.model_card,
    "device": "cuda",
    "lang_arg": args.lang,
    "transcribe_seconds": round(elapsed, 1),
}, indent=1))
print(f"done in {elapsed:.0f}s")
