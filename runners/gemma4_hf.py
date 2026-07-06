# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "transformers>=4.57",
#   "torch>=2.4",
#   "torchvision",
#   "accelerate>=0.33",
#   "soundfile>=0.12",
#   "librosa>=0.10",
#   "pillow>=10",
#   "numpy",
# ]
# ///
# Gemma 4 (E2B/E4B -it) audio ASR via the transformers any-to-any pipeline.
# It's an audio-LLM, not a Whisper-style ASR: chat-prompted, ~30s max audio per
# pass, so we window the audio into <=28s chunks and stitch. CUDA or MPS.
# Latvian audio support is undocumented — this is a measurement, not a bet.
import argparse
import json
import re
import time
from pathlib import Path

import soundfile as sf
from transformers import pipeline

WINDOW = 28.0  # seconds/chunk; model hard-limits audio at ~30s

p = argparse.ArgumentParser()
p.add_argument("--audio", required=True)
p.add_argument("--out", required=True)
p.add_argument("--repo", default="google/gemma-4-E4B-it")
p.add_argument("--language", default="Latvian")
p.add_argument("--max-new-tokens", type=int, default=256)
args = p.parse_args()

pipe = pipeline(task="any-to-any", model=args.repo, device_map="auto", dtype="auto")

audio, sr = sf.read(args.audio, dtype="float32")
if audio.ndim > 1:
    audio = audio.mean(axis=1)

prompt = (
    f"Transcribe the following {args.language} speech segment verbatim. "
    "Output only the transcription text, with no commentary or translation."
)

out = Path(args.out)
tmp = out / "_chunk.wav"
step = int(WINDOW * sr)
n = (len(audio) + step - 1) // step

segs = []
t0 = time.time()
for i in range(n):
    a, b = i * step, min(len(audio), (i + 1) * step)
    sf.write(tmp, audio[a:b], sr)
    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
            {"type": "audio", "audio": str(tmp)},
        ],
    }]
    res = pipe(messages, return_full_text=False,
               generate_kwargs={"max_new_tokens": args.max_new_tokens, "do_sample": False})
    text = res[0]["generated_text"] if res else ""
    if isinstance(text, list):  # chat-style return: take last message's content
        text = text[-1].get("content", "") if text else ""
    text = re.sub(r"<[^>]*turn[^>]*>", " ", str(text))  # strip leaked chat turn tokens
    text = re.sub(r"\s+", " ", text).strip()
    segs.append({"start": round(a / sr, 2), "end": round(b / sr, 2), "text": text})
elapsed = time.time() - t0
tmp.unlink(missing_ok=True)

(out / "transcript.txt").write_text("\n".join(s["text"] for s in segs) + "\n")
(out / "segments.json").write_text(json.dumps(segs, ensure_ascii=False, indent=1))
(out / "meta.json").write_text(json.dumps({
    "backend": "transformers/gemma4",
    "model": args.repo,
    "device": "auto",
    "language_hint": args.language,
    "window_s": WINDOW,
    "chunks": n,
    "transcribe_seconds": round(elapsed, 1),
}, indent=1))
print(f"done in {elapsed:.0f}s, {n} chunks")
