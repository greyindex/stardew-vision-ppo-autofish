"""Build an audio fingerprint from a user-selected bite sound in a recording.

Requires ffmpeg on PATH and NumPy. This writes features, not a playable sound.
The start time is the beginning of the analysis window, slightly before the
first bite pulse. Stop the GUI before replacing its fingerprint.
"""
import argparse
from collections import deque
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--start", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.start < 0 or not args.video.is_file():
        parser.error("Provide an existing recording and a nonnegative start time")
    if args.output.suffix.lower() != ".npz":
        parser.error("Output must have the .npz extension")
    if args.output.exists() or args.output.with_suffix(".json").exists():
        parser.error("Choose a new output path; existing fingerprints and metadata are not overwritten")
    rate, nfft, hop, frames = 24000, 1024, 256, 10
    index = round(args.start*rate/hop)
    end = ((index+frames-1)*hop+nfft)/rate
    decoded = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(args.video),
                              "-t", str(end+.05), "-vn", "-ac", "1", "-ar", str(rate),
                              "-f", "s16le", "pipe:1"], check=True, stdout=subprocess.PIPE).stdout
    audio = np.frombuffer(decoded, np.int16).astype(np.float32)/32768
    if len(audio) < round(end*rate):
        parser.error("Recording ends before the requested fingerprint window")
    noise = np.zeros(153, np.float32)
    history = deque(maxlen=frames)
    window = np.hanning(nfft).astype(np.float32)
    for i in range(index+frames):
        samples = audio[i*hop:i*hop+nfft]
        spectrum = np.abs(np.fft.rfft(samples*window))[7:160]/(nfft/2)
        history.append(np.sqrt(np.maximum(spectrum-.9*noise, 0)))
        noise = .99*noise+.01*spectrum
    template = np.stack(history, axis=1).astype(np.float32)
    if not np.isfinite(template).all() or float(np.linalg.norm(template)) < 1e-5:
        parser.error("The selected window is silent or invalid")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, schema=np.array("stardew-bite-spectrum-v1"),
                        sample_rate=np.int32(rate), nfft=np.int32(nfft), hop=np.int32(hop),
                        bins=np.array([7,160],np.int32), template=template,
                        threshold=np.float32(.93), support_threshold=np.float32(.88))
    metadata = {"schema":"stardew-bite-spectrum-v1", "source": args.video.name,
                "start_s": index*hop/rate, "end_s": end,
                "contents": "Spectral features only; source recording is not copied",
                "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
                "note": "Re-evaluate thresholds against other sounds before deployment."}
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2)+"\n", encoding="utf-8")
    print(f"Saved {template.shape[0]} x {template.shape[1]} spectral features to {args.output}")


if __name__ == "__main__":
    main()
