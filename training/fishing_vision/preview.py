"""Render a checkpoint's predictions on recorded panels; never controls a game."""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw
from .inference import VisionPredictor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--video", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--every", type=float, default=.5)
    parser.add_argument("--max-frames", type=int, default=60)
    args = parser.parse_args()
    torch.set_num_threads(2); cv2.setNumThreads(1)
    model = VisionPredictor(args.checkpoint)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(args.video); fps = cap.get(cv2.CAP_PROP_FPS)
    frames, indices = [], []
    i = 0; stride = max(1, round(fps*args.every))
    while len(frames) < args.max_frames:
        ok, frame = cap.read()
        if not ok: break
        if i % stride == 0: frames.append(frame); indices.append(i)
        i += 1
    cap.release()
    if not frames: raise ValueError("No frames decoded")
    predictions = []
    for start in range(0, len(frames), 8): predictions.extend(model.predict(frames[start:start+8]))
    serialized = []
    selected = list(range(len(frames)))[::max(1, len(frames)//12)][:12]
    sheet = Image.new("RGB", (208*len(selected), 652), "#14202f")
    draw = ImageDraw.Draw(sheet)
    for j, (frame, prediction, index) in enumerate(zip(frames, predictions, indices)):
        record = {k: v for k, v in prediction.items() if k != "mask"}
        record["time_in_clip_s"] = index/fps
        serialized.append(record)
        if j not in selected: continue
        ordinal = selected.index(j)
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).resize((188, 600))
        d = ImageDraw.Draw(image); yscale = 600/frame.shape[0]
        for name, color, presence_key, xs in (
                ("fish_visual_center", "cyan", "fish", (55, 115)),
                ("bar_top", "magenta", "bar", (55, 115)),
                ("bar_bottom", "magenta", "bar", (55, 115)),
                ("progress_top", "orange", "progress", (123, 145))):
            yy = prediction["rows"][name]*yscale
            # Raw estimates are still saved in JSON. Suppress meaningless rows
            # in the contact sheet when that object was predicted absent.
            if (prediction["panel_candidate"]
                    and prediction["presence_scores"][presence_key] >= .7
                    and 0 <= yy < 600):
                d.line((xs[0], yy, xs[1], yy), fill=color, width=2)
        sheet.paste(image, (ordinal*208+10, 48))
        draw.text((ordinal*208+10, 5), f"{index/fps:.2f}s / step {model.step}", fill="white")
        draw.text((ordinal*208+10, 23), f"panel={prediction['presence_scores']['panel']:.2f}", fill="white")
    sheet.save(out/"predictions.jpg", quality=94)
    (out/"predictions.json").write_text(json.dumps({"source": args.video, "checkpoint_step": model.step,
                      "label_status": "unlabelled qualitative replay, not accuracy", "frames": serialized}, indent=2), encoding="utf-8")
    print(json.dumps({"step": model.step, "frames": len(frames), "output": str(out)}))


if __name__ == "__main__": main()
