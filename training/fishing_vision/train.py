"""Bounded, resumable GPU training. Writes progress and checkpoints locally."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import time
import traceback

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler
from . import SCHEMA, WIDTH, HEIGHT, CLASSES, ROWS, PRESENCE
from .model import FishingVision, objective
from .synthetic import SyntheticFishing, LabelledCrops


def atomic_json(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    for attempt in range(10):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            time.sleep(.025 * (attempt+1))
    raise PermissionError(f"Cannot update {path}")


def save_checkpoint(path, data):
    path = Path(path); temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(data, temporary)
    for attempt in range(10):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            time.sleep(.025*(attempt+1))
    raise PermissionError(f"Cannot update {path}")


def device_batch(batch, device):
    result = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
    result["image"] = result["image"].contiguous(memory_format=torch.channels_last)
    return result


@torch.inference_mode()
def validate(model, loader, device):
    model.eval()
    errors = [[] for _ in ROWS]
    count, total_loss, presence_ok, presence_n = 0, 0., 0, 0
    false_positive, negatives = 0, 0
    confusion = torch.zeros(len(CLASSES), len(CLASSES), device=device, dtype=torch.int64)
    for raw in loader:
        batch = device_batch(raw, device)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            pred = model(batch["image"])
            loss, _ = objective(pred, batch)
        n = len(raw["image"]); count += n; total_loss += float(loss)*n
        estimate = pred["rows"].argmax(-1)
        for j in range(len(ROWS)):
            valid = batch["row_valid"][:, j] > .5
            errors[j].extend((estimate[:, j][valid]-batch["rows"][:, j][valid]).abs().float().cpu().tolist())
        labels = pred["seg"].argmax(1)
        target = batch["mask"]
        valid = target >= 0
        pairs = target[valid]*len(CLASSES)+labels[valid]
        confusion += torch.bincount(pairs, minlength=len(CLASSES)**2).reshape(len(CLASSES), len(CLASSES))
        binary = pred["presence"].sigmoid() > .5
        truth = batch["presence"] > .5
        presence_ok += int((binary == truth).sum()); presence_n += truth.numel()
        neg = ~truth[:, 0]; negatives += int(neg.sum()); false_positive += int(binary[:, 0][neg].sum())
    matrix = confusion.cpu().numpy()
    iou = np.diag(matrix) / np.maximum(matrix.sum(0)+matrix.sum(1)-np.diag(matrix), 1)
    return {"split": "synthetic_validation", "samples": count, "loss": total_loss/max(count, 1),
            "row_error_pixels": {name: {"mean": float(np.mean(e)), "p95": float(np.quantile(e, .95)),
                                        "over_12px_fraction": float(np.mean(np.array(e) > 12))}
                                 for name, e in zip(ROWS, errors) if e},
            "iou": dict(zip(CLASSES, map(float, iou))),
            "presence_accuracy": presence_ok/max(presence_n, 1),
            "negative_panel_false_positives": false_positive, "negative_panel_samples": negatives}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets", default="vision_data/assets")
    parser.add_argument("--negatives", default="vision_data/negatives")
    parser.add_argument("--run", required=True)
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--val-samples", type=int, default=256)
    parser.add_argument("--seed", type=int, default=92701)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--resume")
    parser.add_argument("--real-manifest")
    args = parser.parse_args()
    run = Path(args.run); run.mkdir(parents=True, exist_ok=True)
    if (run/"manifest.json").exists() and not args.resume:
        raise FileExistsError("Use a new run directory or --resume; existing runs are preserved.")
    cv2.setNumThreads(1); torch.set_num_threads(2)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("This run requires CUDA. Use the dedicated .venv-vision runtime.")
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")
    torch.cuda.set_per_process_memory_fraction(.55)
    model = FishingVision().to(device, memory_format=torch.channels_last)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda")
    start_step, best = 0, float("inf")
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=True)
        if checkpoint["schema"] != SCHEMA:
            raise ValueError("Checkpoint schema mismatch")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_step, best = checkpoint["step"], checkpoint.get("best_score", best)
    dataset = SyntheticFishing(args.assets, args.negatives, seed=args.seed)
    validation = SyntheticFishing(args.assets, args.negatives, length=args.val_samples, seed=90000000)
    sampler = RandomSampler(dataset, replacement=True, num_samples=max(1, args.steps-start_step)*args.batch_size,
                            generator=torch.Generator().manual_seed(args.seed+start_step))
    loader = DataLoader(dataset, sampler=sampler, batch_size=args.batch_size, num_workers=args.workers,
                        pin_memory=True, persistent_workers=args.workers > 0)
    val_loader = DataLoader(validation, batch_size=args.batch_size, num_workers=0, pin_memory=True)
    real_loader = None
    if args.real_manifest:
        real = LabelledCrops(args.real_manifest)
        if not len(real):
            raise ValueError("No reviewed training crops; unreviewed pseudo-labels are not accepted")
        real_loader = DataLoader(real, batch_size=max(1, args.batch_size//2), shuffle=True, num_workers=0)
        real_iterator = iter(real_loader)
    sources = {}
    for file in Path(__file__).parent.glob("*.py"):
        sources[file.name] = hashlib.sha256(file.read_bytes()).hexdigest()
    manifest = {"schema": SCHEMA, "args": vars(args), "input": [3, HEIGHT, WIDTH], "classes": CLASSES,
                "rows": ROWS, "presence": PRESENCE, "channels": model.channels, "attention_layers": 0,
                "parameters": sum(p.numel() for p in model.parameters()), "torch": str(torch.__version__),
                "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(), "python": platform.python_version(),
                "source_hashes": sources, "pid": os.getpid(), "label_source": "renderer ground truth + reviewed real annotations only",
                "real_training": bool(args.real_manifest), "legendary_status": "screenshot-derived prototype; no real legendary validation"}
    atomic_json(run/"manifest.json", manifest)
    started = time.monotonic(); step = start_step
    metric_file = (run/"metrics.jsonl").open("a", encoding="utf-8")
    print(json.dumps({"event": "TRAINING_STARTED", **manifest}, ensure_ascii=False), flush=True)
    try:
        for step, raw in enumerate(loader, start_step+1):
            model.train()
            if real_loader is not None:
                try:
                    extra = next(real_iterator)
                except StopIteration:
                    real_iterator = iter(real_loader); extra = next(real_iterator)
                n = len(extra["image"])
                raw = {k: torch.cat((v[:args.batch_size-n], extra[k]), 0) for k, v in raw.items()}
            batch = device_batch(raw, device)
            optimizer.zero_grad(set_to_none=True)
            # Warmup then cosine decrease; includes the absolute resumed step.
            fraction = step / max(args.steps, 1)
            lr = args.lr * min(1., step/100) * (.1 + .9*(1+np.cos(np.pi*fraction))/2)
            for group in optimizer.param_groups: group["lr"] = float(lr)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                pred = model(batch["image"]); loss, parts = objective(pred, batch)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite loss at step {step}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            scaler.step(optimizer); scaler.update()
            if step == start_step+1 or step % 25 == 0:
                elapsed = time.monotonic()-started
                progress = {"state": "training", "pid": os.getpid(), "step": step, "target_steps": args.steps,
                            "samples_seen_this_process": (step-start_step)*args.batch_size,
                            "loss": float(loss), "elapsed_seconds": elapsed,
                            "steps_per_second": (step-start_step)/max(elapsed, 1e-6), "lr": float(lr),
                            "gpu_peak_mib": torch.cuda.max_memory_allocated()/2**20,
                            "best_synthetic_score": best if np.isfinite(best) else None}
                atomic_json(run/"status.json", progress)
                metric_file.write(json.dumps(progress)+"\n"); metric_file.flush()
                print(json.dumps(progress), flush=True)
            if step % args.eval_every == 0 or step == args.steps:
                report = validate(model, val_loader, device)
                report["step"] = step
                score = float(np.mean([v["p95"] for v in report["row_error_pixels"].values()])) + 100*report["negative_panel_false_positives"]/max(1, report["negative_panel_samples"])
                report["selection_score"] = score
                improved = score < best
                best = min(best, score)
                checkpoint = {"schema": SCHEMA, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                              "scaler": scaler.state_dict(), "step": step, "best_score": best,
                              "channels": model.channels, "input": [3, HEIGHT, WIDTH], "validation": report}
                save_checkpoint(run/"latest.pt", checkpoint)
                if improved: save_checkpoint(run/"best.pt", checkpoint)
                atomic_json(run/f"validation_{step:06d}.json", report)
                print(json.dumps({"event": "VALIDATION", **report}), flush=True)
            if step >= args.steps: break
        atomic_json(run/"status.json", {"state": "complete", "pid": os.getpid(), "step": step,
                    "target_steps": args.steps, "elapsed_seconds": time.monotonic()-started,
                    "best_synthetic_score": best, "real_game_verified": False})
        print("TRAINING_COMPLETE", flush=True)
    except BaseException:
        atomic_json(run/"status.json", {"state": "failed", "step": step, "pid": os.getpid(),
                    "error": traceback.format_exc(), "elapsed_seconds": time.monotonic()-started})
        raise
    finally:
        metric_file.close()


if __name__ == "__main__":
    main()
