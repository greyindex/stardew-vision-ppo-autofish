"""CPU PPO training with a curriculum, held-out evaluation and resumable checkpoints."""
import argparse
from collections import deque
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import time
import traceback
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from .env import FishingEnv
from .evaluate import DEVELOPMENT_SEED, evaluate
from .observations import SCHEMA, OBS_SIZE, FRAME_FIELDS, CONFIG_FIELDS, HISTORY

ROOT = Path(__file__).resolve().parent.parent


def replace_with_retry(temp, path):
    # Windows readers and antivirus scans can briefly deny replacement of an open file.
    for attempt in range(40):
        try:
            os.replace(temp, path)
            return
        except PermissionError:
            if attempt == 39:
                raise
            time.sleep(0.05)


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    replace_with_retry(temp, path)


def save_model(model, path):
    path = Path(path)
    temp = path.with_name(path.stem + ".writing.zip")
    model.save(temp)
    replace_with_retry(temp, path)


class TrainingProgress(BaseCallback):
    def __init__(self, run, total, eval_every, eval_episodes, curriculum=True,
                 eval_seed=DEVELOPMENT_SEED, eval_at_start=False, save_checkpoints=False,
                 recovery=None):
        super().__init__()
        self.run, self.total = run, total
        self.eval_every, self.eval_episodes = eval_every, eval_episodes
        self.curriculum = curriculum
        self.eval_seed = eval_seed
        self.eval_at_start = eval_at_start
        self.save_checkpoints = save_checkpoints
        self.recent = deque(maxlen=200)
        self.episodes = 0
        self.last_write = 0
        self.last_eval = 0
        self.best = -1.0
        self.latest_eval = None
        self.started = time.monotonic()
        self.stage = "easy" if curriculum else "full"
        self.start_steps = 0
        self.recovery = recovery or {}
        self.elapsed_offset = self.recovery.get("previous_elapsed_seconds", 0.0)
        if recovery:
            self.episodes = recovery["previous_observed_episodes"]
            selection = json.loads((run / "best_selection.json").read_text(encoding="utf-8"))
            self.best = selection["catch_rate"]
            checkpoint_eval = run / f"eval_{recovery['resume_steps']:09d}.json"
            result = json.loads(checkpoint_eval.read_text(encoding="utf-8"))
            self.latest_eval = {key: value for key, value in result.items() if key != "episodes_detail"}

    def _on_training_start(self):
        self.start_steps = self.recovery.get("original_start_steps", self.model.num_timesteps)
        self.last_eval = self.model.num_timesteps
        self.write("training")
        if self.eval_at_start:
            self.run_evaluation()

    def run_evaluation(self):
        self.write("evaluating")
        save_model(self.model, self.run / "latest.zip")
        if self.save_checkpoints:
            save_model(self.model, self.run / f"checkpoint_{self.model.num_timesteps:09d}.zip")
        result = evaluate(model=self.model, episodes=self.eval_episodes, seed=self.eval_seed, stage="full")
        result["steps"] = self.model.num_timesteps
        result["seed"] = self.eval_seed
        atomic_json(self.run / f"eval_{self.model.num_timesteps:09d}.json", result)
        self.latest_eval = {key: value for key, value in result.items() if key != "episodes_detail"}
        if result["catch_rate"] > self.best:
            self.best = result["catch_rate"]
            save_model(self.model, self.run / "best.zip")
            atomic_json(self.run / "best_selection.json", {
                "steps": self.model.num_timesteps, "seed": self.eval_seed,
                "episodes": self.eval_episodes, "catch_rate": self.best,
                "selection_rule": "highest development catch rate; earlier checkpoint wins ties",
            })
        self.last_eval = self.model.num_timesteps
        print(f"Eval {self.model.num_timesteps:,}: catch={result['catch_rate']:.2%}, perfect={result['perfect_rate']:.2%}", flush=True)
        self.write("training")

    def write(self, state, error=None):
        elapsed = max(self.elapsed_offset + time.monotonic() - self.started, 0.001)
        recent = list(self.recent)
        stats = {
            "state": state, "pid": os.getpid(), "steps": self.model.num_timesteps,
            "target_steps": self.total, "stage": self.stage, "episodes": self.episodes,
            "elapsed_seconds": elapsed, "steps_per_second": (self.model.num_timesteps - self.start_steps) / elapsed,
            "train_recent_catch_rate": sum(r["caught"] for r in recent) / len(recent) if recent else None,
            "train_recent_mean_seconds": sum(r["seconds"] for r in recent) / len(recent) if recent else None,
            "best_eval_catch_rate": self.best if self.best >= 0 else None,
            "latest_evaluation": self.latest_eval,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "run_dir": str(self.run), "error": error,
        }
        try:
            atomic_json(self.run / "status.json", stats)
            with (self.run / "metrics.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(stats, ensure_ascii=False) + "\n")
        except OSError as exc:
            # Telemetry must not discard an in-memory training run.
            print(f"Status write delayed ({state} at {self.model.num_timesteps}): {exc}", flush=True)
        self.last_write = time.monotonic()

    def _on_step(self):
        for info, done in zip(self.locals["infos"], self.locals["dones"]):
            if done:
                self.episodes += 1
                self.recent.append(info)
        fraction = self.model.num_timesteps / self.total
        next_stage = "easy" if fraction < 0.125 else "mixed" if fraction < 0.5 else "full"
        if not self.curriculum:
            next_stage = "full"
        if next_stage != self.stage:
            self.training_env.env_method("set_stage", next_stage)
            self.stage = next_stage
            self.recent.clear()
            print(f"Curriculum -> {self.stage} at {self.model.num_timesteps:,} decisions", flush=True)
        if self.n_calls % 64 == 0:
            if (self.run / "STOP").exists():
                self.write("stopping")
                return False
            if time.monotonic() - self.last_write >= 2:
                self.write("training")
        if self.model.num_timesteps - self.last_eval >= self.eval_every:
            self.run_evaluation()
        return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=2_000_000)
    parser.add_argument("--envs", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--eval-every", type=int, default=200_000)
    parser.add_argument("--eval-episodes", type=int, default=80)
    parser.add_argument("--eval-seed", type=int, default=DEVELOPMENT_SEED)
    parser.add_argument("--eval-at-start", action="store_true")
    parser.add_argument("--eval-at-end", action="store_true")
    parser.add_argument("--save-checkpoints", action="store_true", help="Retain weights at every evaluation point")
    parser.add_argument("--run-dir", default=str(ROOT / "runs" / datetime.now().strftime("%Y%m%d_%H%M%S")))
    parser.add_argument("--resume", help="Resume optimizer and weights from a checkpoint into a new run")
    parser.add_argument("--continue-run", action="store_true", help="Recover an interrupted run using its recovery.json record")
    parser.add_argument("--no-curriculum", action="store_true")
    args = parser.parse_args()
    if args.steps < 1 or not 1 <= args.envs <= 64 or not 1 <= args.threads <= 16 or args.eval_every < 1 or args.eval_episodes < 1:
        parser.error("Invalid training budget or parallelism")
    run = Path(args.run_dir).resolve()
    run.mkdir(parents=True, exist_ok=True)
    prior_manifest = None
    recovery = None
    if args.continue_run:
        if not args.resume or not args.no_curriculum:
            parser.error("Recovery requires --resume and a completed curriculum (--no-curriculum)")
        prior_manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
        recovery = json.loads((run / "recovery.json").read_text(encoding="utf-8-sig"))
        for key in ("envs", "seed", "threads", "eval_every", "eval_episodes", "eval_seed", "no_curriculum"):
            if vars(args)[key] != prior_manifest["arguments"][key]:
                parser.error(f"Recovery cannot change {key}")
        if Path(args.resume).resolve().parent != run or args.eval_at_start:
            parser.error("Recover from a checkpoint in this run, without repeating the starting evaluation")
    elif (run / "manifest.json").exists():
        parser.error("Choose a new run directory; existing experiments are never overwritten")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    stage = "full" if args.no_curriculum else "easy"
    vec = DummyVecEnv([lambda: Monitor(FishingEnv(stage=stage)) for _ in range(args.envs)])
    vec.seed(args.seed)
    model = None
    callback = None
    try:
        if args.resume:
            model = PPO.load(args.resume, env=vec, device="cpu")
        else:
            model = PPO(
                "MlpPolicy", vec, policy_kwargs={"net_arch": {"pi": [128, 128], "vf": [128, 128]}, "activation_fn": torch.nn.Tanh},
                learning_rate=3e-4, n_steps=256, batch_size=512, n_epochs=5,
                gamma=1.0, gae_lambda=0.95, clip_range=0.2, ent_coef=0.01,
                vf_coef=0.5, max_grad_norm=0.5, target_kl=0.02,
                device="cpu", seed=args.seed, verbose=0,
            )
        total = model.num_timesteps + args.steps
        if recovery and (model.num_timesteps != recovery["resume_steps"] or total != prior_manifest["target_steps"]):
            raise ValueError("Recovery checkpoint or total budget does not match the recorded run")
        callback = TrainingProgress(run, total, args.eval_every, args.eval_episodes, not args.no_curriculum,
                                    args.eval_seed, args.eval_at_start, args.save_checkpoints, recovery)
        manifest = {
            "arguments": vars(args), "schema": SCHEMA, "observation_size": OBS_SIZE,
            "history": HISTORY, "frame_fields": FRAME_FIELDS, "config_fields": CONFIG_FIELDS,
            "python": sys.version, "platform": platform.platform(),
            "packages": {name: importlib.metadata.version(name) for name in ("numpy", "gymnasium", "stable-baselines3", "torch")},
            "source_revision": "506322f56b2ae1975e0c896fe7bccb731e698fac",
            "training_seed": args.seed, "development_seed": args.eval_seed,
            "start_steps": model.num_timesteps, "target_steps": total,
            "device": "cpu", "objective": "catch-only", "policy": "MLP pi=[128,128] vf=[128,128]",
            "note": "Simulation-only performance; exact geometry, no vision or latency noise. Catalogue and dynamics are Pufferdle reference, not a current-game certification.",
        }
        if prior_manifest is not None:
            manifest = {**prior_manifest, "resumptions": [*prior_manifest.get("resumptions", []),
                        {"arguments": vars(args), "recovery": recovery, "at": datetime.now(timezone.utc).isoformat()}]}
        atomic_json(run / "manifest.json", manifest)
        print(f"Run: {run}\nPPO CPU | {args.envs} envs | {OBS_SIZE} observations | {total:,} target decisions", flush=True)
        if not args.continue_run:
            save_model(model, run / "initial.zip")
        model.learn(total_timesteps=args.steps, callback=callback, reset_num_timesteps=not bool(args.resume))
        save_model(model, run / "latest.zip")
        save_model(model, run / "final.zip")
        stopped = (run / "STOP").exists()
        if args.eval_at_end and not stopped:
            callback.run_evaluation()
        callback.write("stopped" if stopped else "completed")
        print("Stopped safely; checkpoint saved." if stopped else "Training completed; final checkpoint saved.", flush=True)
    except KeyboardInterrupt:
        if model is not None:
            save_model(model, run / "latest.zip")
        if callback is not None:
            callback.write("stopped")
    except Exception:
        error = traceback.format_exc()
        if callback is not None:
            callback.write("failed", error)
        else:
            atomic_json(run / "status.json", {"state": "failed", "error": error})
        raise
    finally:
        vec.close()


if __name__ == "__main__":
    main()
