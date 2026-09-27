"""Fixed seed evaluation; identities are for reports, never policy inputs."""
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import time
import numpy as np
from .controllers import MPC, PredictiveRule
from .env import FishingEnv

DEVELOPMENT_SEED = 1_000_000
FINAL_SEED = 2_000_000


def evaluate(policy=None, episodes=100, seed=DEVELOPMENT_SEED, stage="full", model=None, level=None):
    # Batch policy inference for PPO. Baselines use the identical observation contract.
    width = min(16 if model is not None else 1, episodes)
    envs = [FishingEnv(stage=stage, level=level) for _ in range(width)]
    active, completed = {}, []
    next_episode = 0
    for i, env in enumerate(envs):
        obs, _ = env.reset(seed=seed + next_episode)
        active[i] = (obs, seed + next_episode)
        next_episode += 1
    started = time.monotonic()
    while active:
        keys = list(active)
        batch = np.stack([active[k][0] for k in keys])
        if model is not None:
            actions = model.predict(batch, deterministic=True)[0]
        else:
            actions = [policy(obs) for obs in batch]
        for key, action in zip(keys, actions):
            obs, _, terminal, _, info = envs[key].step(int(action))
            episode_seed = active[key][1]
            if terminal:
                info["seed"] = episode_seed
                completed.append(info)
                if next_episode < episodes:
                    obs, _ = envs[key].reset(seed=seed + next_episode)
                    active[key] = (obs, seed + next_episode)
                    next_episode += 1
                else:
                    del active[key]
            else:
                active[key] = (obs, episode_seed)
    for env in envs:
        env.close()
    return summarize(completed, stage, time.monotonic() - started)


def summarize(rows, stage, wall_seconds):
    def stats(items):
        n = len(items)
        p = sum(row["caught"] for row in items) / n
        z = 1.96
        center = (p + z*z/(2*n)) / (1 + z*z/n)
        half = z * ((p*(1-p)/n + z*z/(4*n*n)) ** 0.5) / (1 + z*z/n)
        caught = [row["seconds"] for row in items if row["caught"]]
        return {
            "episodes": n, "catch_rate": p, "catch_ci95": [center-half, center+half],
            "perfect_rate": sum(row["perfect"] for row in items) / n,
            "mean_seconds": sum(row["seconds"] for row in items) / n,
            "mean_catch_seconds": float(np.mean(caught)) if caught else None,
            "timeout_rate": sum(row["outcome"] == "timeout" for row in items) / n,
        }
    groups = defaultdict(list)
    difficulties = defaultdict(list)
    for row in rows:
        groups[row["behavior"]].append(row)
        d = row["difficulty"]
        difficulties["<40" if d < 40 else "40-69" if d < 70 else "70-89" if d < 90 else ">=90"].append(row)
    return {**stats(rows), "stage": stage, "wall_seconds": wall_seconds,
            "by_behavior": {key: stats(value) for key, value in groups.items()},
            "by_difficulty": {key: stats(value) for key, value in difficulties.items()},
            "episodes_detail": rows}


def baseline_worker(controller, episodes, seed, stage, level):
    rng = np.random.default_rng(seed + 47)
    policy = MPC() if controller == "mpc" else PredictiveRule() if controller == "rule" else lambda obs: int(rng.integers(2))
    return evaluate(policy=policy, episodes=episodes, seed=seed, stage=stage, level=level)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--controller", choices=["rule", "mpc", "ppo", "random"], default="rule")
    parser.add_argument("--model")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--seed", type=int, default=FINAL_SEED)
    parser.add_argument("--stage", choices=["easy", "mixed", "full"], default="full")
    parser.add_argument("--level", type=int)
    parser.add_argument("--workers", type=int, default=1, help="Parallel processes for CPU baselines")
    parser.add_argument("--output", default="runs/evaluation.json")
    args = parser.parse_args()
    if args.episodes < 1 or not 1 <= args.workers <= 16:
        parser.error("episodes must be positive")
    model = None
    rng = np.random.default_rng(2026)
    policy = PredictiveRule() if args.controller == "rule" else MPC() if args.controller == "mpc" else lambda obs: int(rng.integers(2))
    if args.controller == "ppo":
        if not args.model:
            parser.error("--model is required for PPO")
        import torch
        from stable_baselines3 import PPO
        torch.set_num_threads(2)
        model = PPO.load(args.model, device="cpu")
    if args.workers > 1 and model is None:
        started = time.monotonic()
        workers = min(args.workers, args.episodes)
        counts = [args.episodes // workers + int(i < args.episodes % workers) for i in range(workers)]
        offset = 0
        with ProcessPoolExecutor(max_workers=workers) as pool:
            jobs = []
            for count in counts:
                jobs.append(pool.submit(baseline_worker, args.controller, count, args.seed + offset, args.stage, args.level))
                offset += count
            rows = [row for job in jobs for row in job.result()["episodes_detail"]]
        result = summarize(rows, args.stage, time.monotonic() - started)
    else:
        result = evaluate(policy=policy, model=model, episodes=args.episodes, seed=args.seed, stage=args.stage, level=args.level)
    result["controller"] = args.controller
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k:v for k,v in result.items() if k != "episodes_detail"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
