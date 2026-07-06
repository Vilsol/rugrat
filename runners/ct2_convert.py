# /// script
# requires-python = ">=3.10"
# dependencies = ["ctranslate2>=4.5.0", "transformers>=4.40", "torch", "huggingface_hub"]
#
# [tool.uv.sources]
# torch = { index = "pytorch-cpu" }
#
# [[tool.uv.index]]
# name = "pytorch-cpu"
# url = "https://download.pytorch.org/whl/cpu"
# explicit = true
# ///
# One-time: build a CTranslate2 model from an HF Whisper checkpoint at a chosen
# precision. The AiLab cv19 repo only ships an int8 CT2 export; convert its
# root safetensors to float16 so faster-whisper decodes at full precision.
# Conversion is CPU-only (weights are just re-serialized), hence CPU torch.
import argparse
import sys
from pathlib import Path

from huggingface_hub import snapshot_download
from ctranslate2.converters import TransformersConverter

p = argparse.ArgumentParser()
p.add_argument("--repo", required=True)
p.add_argument("--out", required=True)
p.add_argument("--quant", default="float16")
args = p.parse_args()

out = Path(args.out)
if (out / "model.bin").exists():
    print(f"[convert] reuse {out}", file=sys.stderr)
    sys.exit(0)

src = snapshot_download(args.repo, allow_patterns=[
    "*.safetensors", "config.json", "generation_config.json", "tokenizer.json",
    "vocab.json", "merges.txt", "normalizer.json", "added_tokens.json",
    "special_tokens_map.json", "tokenizer_config.json", "preprocessor_config.json",
])
TransformersConverter(src, copy_files=["tokenizer.json", "preprocessor_config.json"]).convert(
    str(out), quantization=args.quant, force=True)
print(f"[convert] wrote {out} ({args.quant})", file=sys.stderr)
