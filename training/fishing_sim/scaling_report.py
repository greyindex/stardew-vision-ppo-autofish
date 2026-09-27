"""Evaluate a completed continuation on its predeclared holdout and report scaling.

This is an analysis utility. It never trains or changes model selection.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from .evaluate import evaluate


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def weight_digest(model):
    digest = hashlib.sha256()
    for name, tensor in sorted(model.policy.state_dict().items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def paired_difference(before, after, hard_only=False):
    """Episode-paired percentile bootstrap, conditioned on this one trained policy pair."""
    left = {row["seed"]: row for row in before["episodes_detail"]}
    right = {row["seed"]: row for row in after["episodes_detail"]}
    if left.keys() != right.keys():
        raise ValueError("Paired evaluation requires identical episode seeds")
    differences = []
    for seed in sorted(left):
        a, b = left[seed], right[seed]
        if any(a[field] != b[field] for field in ("fish", "difficulty", "behavior", "level", "tackle")):
            raise ValueError(f"Mismatched episode configuration at seed {seed}")
        if hard_only and a["difficulty"] < 90:
            continue
        differences.append(int(b["caught"]) - int(a["caught"]))
    values = np.asarray(differences)
    counts = np.asarray([(values == -1).sum(), (values == 0).sum(), (values == 1).sum()])
    n = len(values)
    if not n:
        raise ValueError("Empty paired comparison")
    rng = np.random.default_rng(20260927)
    # Multinomial draws are exactly the empirical bootstrap of {-1, 0, +1} outcomes.
    samples = rng.multinomial(n, counts / n, size=20000)
    deltas = (samples[:, 2] - samples[:, 0]) / n
    low, high = np.quantile(deltas, [0.025, 0.975])
    return {
        "episodes": n, "gain_pp": float(values.mean() * 100),
        "gain_ci95_pp": [float(low * 100), float(high * 100)],
        "improved_episodes": int(counts[2]), "regressed_episodes": int(counts[0]),
        "unchanged_episodes": int(counts[1]),
        "method": "20,000 episode-paired bootstrap draws; does not measure training-seed variability",
    }


def evaluate_holdout(run, plan):
    import torch
    from stable_baselines3 import PPO
    torch.set_num_threads(2)
    spec = plan["final_holdout"]
    records = {}
    cache = {}
    for label, name in (("start", "initial.zip"), ("terminal", "final.zip"), ("selected", "best.zip")):
        model = PPO.load(run / name, device="cpu")
        digest = weight_digest(model)
        output = run / f"holdout_{label}.json"
        existing = read_json(output) if output.exists() else None
        if existing and (existing.get("weight_sha256"), existing.get("seed"), existing.get("episodes")) == (digest, spec["seed_start"], spec["episodes"]):
            result = existing
        elif digest in cache:
            result = dict(cache[digest])
        else:
            print(f"Holdout {label}: {model.num_timesteps:,} decisions, {spec['episodes']:,} episodes", flush=True)
            result = evaluate(model=model, episodes=spec["episodes"], seed=spec["seed_start"], stage="full")
        result = {**result, "controller": "ppo", "steps": int(model.num_timesteps),
                  "model": str(run / name), "seed": spec["seed_start"], "weight_sha256": digest,
                  "selection": "Development catch rate only; no holdout model selection"}
        write_json(output, result)
        cache[digest] = result
        records[label] = result
        print(f"Holdout {label}: catch={result['catch_rate']:.2%}, hard={result['by_difficulty']['>=90']['catch_rate']:.2%}", flush=True)
    # The dashboard loads best.zip, so its test card must describe the selected model.
    write_json(run / "final_evaluation.json", records["selected"])
    return records


def make_report(run):
    plan = read_json(run / "experiment_plan.json")
    curve = sorted((read_json(path) for path in run.glob("eval_*.json")), key=lambda item: item["steps"])
    if len(curve) < 2:
        raise ValueError("Need at least two evaluation points")
    for point in curve:
        if (point["seed"], point["episodes"]) != (plan["development"]["seed_start"], plan["development"]["episodes"]):
            raise ValueError("Development pool changed during the experiment")
    held = {label: read_json(run / f"holdout_{label}.json") for label in ("start", "terminal", "selected")}
    selection = read_json(run / "best_selection.json")
    status = read_json(run / "status.json")
    recovery = read_json(run / "recovery.json") if (run / "recovery.json").exists() else None
    rows = []
    for point in curve:
        hard = point["by_difficulty"][">=90"]
        gain = paired_difference(curve[0], point)
        rows.append({
            "steps": point["steps"], "catch_pct": point["catch_rate"] * 100,
            "catch_ci95_low_pct": point["catch_ci95"][0] * 100,
            "catch_ci95_high_pct": point["catch_ci95"][1] * 100,
            "hard_catch_pct": hard["catch_rate"] * 100,
            "hard_episodes": hard["episodes"], "perfect_pct": point["perfect_rate"] * 100,
            "mean_catch_seconds": point["mean_catch_seconds"],
            "gain_from_start_pp": gain["gain_pp"],
            "gain_ci95_low_pp": gain["gain_ci95_pp"][0], "gain_ci95_high_pp": gain["gain_ci95_pp"][1],
        })
    with (run / "scaling_curve.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    comparisons = {}
    for name in ("terminal", "selected"):
        comparisons[name] = {
            "overall": paired_difference(held["start"], held[name]),
            "hard": paired_difference(held["start"], held[name], hard_only=True),
        }
    levels = []
    for level in range(11):
        row = {"level": level}
        for label, result in held.items():
            subset = [item for item in result["episodes_detail"] if item["level"] == level]
            row[label] = {"episodes": len(subset),
                          "catch_rate": sum(item["caught"] for item in subset) / len(subset)}
        levels.append(row)
    intervals = []
    for start, stop in ((2, 4), (4, 6), (6, 8), (8, 10), (6, 10)):
        a = min(curve, key=lambda p: abs(p["steps"] - start * 1_000_000))
        b = min(curve, key=lambda p: abs(p["steps"] - stop * 1_000_000))
        intervals.append({"from_steps": a["steps"], "to_steps": b["steps"],
                          "overall": paired_difference(a, b),
                          "hard": paired_difference(a, b, hard_only=True)})
    summary = {
        "start_steps": curve[0]["steps"], "terminal_steps": held["terminal"]["steps"],
        "best_development_steps": selection["steps"],
        "continuation_wall_seconds_including_development": status["elapsed_seconds"],
        "development_curve": rows, "development_intervals": intervals,
        "holdout": {label: {key: value for key, value in result.items() if key != "episodes_detail"}
                    for label, result in held.items()},
        "holdout_paired_comparisons": comparisons, "holdout_by_level": levels,
        "limitations": plan["limitations"],
    }
    if recovery:
        summary["recovery"] = recovery
        summary["additional_collected_decisions_including_discarded"] = (
            summary["terminal_steps"] - summary["start_steps"] + recovery["discarded_decisions"])
        summary["limitations"] = [*summary["limitations"], recovery["limitation"]]
    write_json(run / "scaling_summary.json", summary)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), layout="constrained")
    x = np.asarray([p["steps"] / 1e6 for p in curve])
    for ax, group, title, color in ((axes[0], None, f"All fish (n={curve[0]['episodes']:,})", "#187e62"),
                                     (axes[1], ">=90", f"Difficulty >= 90 (n={curve[0]['by_difficulty']['>=90']['episodes']})", "#c27224")):
        stats = [p if group is None else p["by_difficulty"][group] for p in curve]
        rates = np.asarray([p["catch_rate"] * 100 for p in stats])
        bounds = np.asarray([p["catch_ci95"] for p in stats]) * 100
        ax.fill_between(x, bounds[:, 0], bounds[:, 1], color=color, alpha=0.12, label="95% Wilson interval")
        ax.plot(x, rates, "o-", color=color, linewidth=2, label="Catch rate")
        ax.set(title=title, xlabel="Total training decisions (millions)", ylabel="Catch success (%)")
        ax.set_xticks(x[::2], [f"{value:.0f}" for value in x[::2]])
        if recovery:
            ax.axvline(recovery["resume_steps"] / 1e6, color="#888888", linewidth=1,
                       linestyle=":", label="Checkpoint recovery")
        ax.grid(axis="y", alpha=0.2)
        ax.legend(loc="best", frameon=False, fontsize=9)
    fig.suptitle("PPO scaling: fixed development set, one training seed, 30 Hz decisions", fontsize=13)
    fig.savefig(run / "scaling_curve.png", dpi=180)
    fig.savefig(run / "scaling_curve.svg")
    plt.close(fig)

    lines = ["# 单纯增加训练步数：200 万到 1000 万", "",
             "保持网络、奖励、学习率、采样与控制频率；原有课程已完成，续训始终使用完整鱼池。", "",
             "## 固定开发集的学习曲线", "",
             f"每个节点使用相同 {plan['development']['episodes']:,} 个种子。难鱼是难度 ≥90 的固定子集。", "",
             "| 累计决策 | 捕获率 | 难鱼捕获率 | 完美捕获/总尝试 |", "|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['steps']:,} | {row['catch_pct']:.2f}% | {row['hard_catch_pct']:.2f}% | {row['perfect_pct']:.2f}% |")
    lines += ["", "![学习曲线](scaling_curve.png)", "", "## 独立保留测试", "",
              f"相同 {plan['final_holdout']['episodes']:,} 个新种子，仅在训练结束后评估。检查点选择只使用开发集。", "",
              "| 模型 | 决策数 | 捕获率 | 95% Wilson 区间 | 难鱼捕获率 | 完美率 |",
              "|---|---:|---:|---:|---:|---:|"]
    for label, title in (("start", "续训起点"), ("terminal", "1000 万步终点"), ("selected", "开发集选出的模型")):
        result = held[label]
        lo, hi = result["catch_ci95"]
        lines.append(f"| {title} | {result['steps']:,} | {result['catch_rate']:.2%} | {lo:.2%}–{hi:.2%} | {result['by_difficulty']['>=90']['catch_rate']:.2%} | {result['perfect_rate']:.2%} |")
    lines += ["", "## 配对比较", "", "| 对比 | 总体增益（百分点） | 95% 配对区间 | 新增成功 / 退步 |", "|---|---:|---:|---:|"]
    for name, title in (("terminal", "终点相对起点"), ("selected", "开发集所选模型相对起点")):
        result = comparisons[name]["overall"]
        lo, hi = result["gain_ci95_pp"]
        lines.append(f"| {title} | {result['gain_pp']:+.2f} | {lo:+.2f}–{hi:+.2f} | {result['improved_episodes']} / {result['regressed_episodes']} |")
    lines += ["", "## 不同钓鱼等级（保留测试，均无渔具）", "",
              "| 等级 | 次数 | 起点捕获率 | 终点捕获率 | 开发集所选模型捕获率 |", "|---:|---:|---:|---:|---:|"]
    for row in levels:
        lines.append(f"| {row['level']} | {row['start']['episodes']} | {row['start']['catch_rate']:.2%} | {row['terminal']['catch_rate']:.2%} | {row['selected']['catch_rate']:.2%} |")
    lines += ["", "## 后期边际收益（开发集）", "", "| 累计步数区间 | 总体增益（百分点） | 95% 配对区间 |", "|---|---:|---:|"]
    for interval in intervals:
        result = interval["overall"]
        lo, hi = result["gain_ci95_pp"]
        lines.append(f"| {interval['from_steps'] / 1e6:.1f}M → {interval['to_steps'] / 1e6:.1f}M | {result['gain_pp']:+.2f} | {lo:+.2f}–{hi:+.2f} |")
    lines += ["", "## 实验边界与存档", "",
              f"- 续训进程及开发集评估累计运行 {status['elapsed_seconds'] / 60:.2f} 分钟；不含修复停顿与保留测试。",
              "- 单一训练种子；区间描述测试轨迹抽样不确定性，不覆盖训练种子差异。",
              "- 恢复权重与优化器，环境轨迹和随机流重新开始；不是逐位恢复此前运行。",
              "- 原课程保持已完成状态。若从零按 1000 万总步数重新拉长课程，会是另一项实验。",
              "- 训练按完整 rollout 更新，因此实际决策数略高于目标；中间检查点取自采样回调，终点在最后一次优化器更新后保存。",
              "- 全部结果来自 Pufferdle 数值模拟；没有视觉误差和真实游戏延迟。",
              "- 预先指定的实验条件见 experiment_plan.json；完整指标见 scaling_summary.json；逐回合结果见 holdout_*.json。",
              "- 训练器 CLI 的 --steps 是新增决策数，本轮请求新增 7,997,056 步，使累计目标为 10,000,000。", ""]
    if recovery:
        lines += ["## 中断与恢复记录", "",
                  f"训练曾在 {recovery['failed_observed_steps']:,} 步时因 Windows 拒绝替换状态文件退出。加入文件替换重试和非致命状态写入后，从 {recovery['resume_steps']:,} 步的已保存权重与优化器恢复。",
                  "",
                  f"回退丢弃的 {recovery['discarded_decisions']:,} 次决策不计入最终模型步数，但耗时计入运行成本。恢复重新开始随机流与环境轨迹，因此不是一次完全不中断的续训。网络、奖励、采样、学习率与动作频率没有改变。",
                  "",
                  "原始错误与状态保留在 interruption_01.*；恢复条件、代码和检查点哈希保留在 recovery.json。", ""]
    (run / "SCALING_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--evaluate-holdout", action="store_true")
    args = parser.parse_args()
    run = Path(args.run_dir).resolve()
    status = read_json(run / "status.json")
    if status["state"] != "completed":
        parser.error("Training must be completed before evaluating the final holdout")
    if args.evaluate_holdout:
        evaluate_holdout(run, read_json(run / "experiment_plan.json"))
    summary = make_report(run)
    print(json.dumps({key: summary[key] for key in ("start_steps", "terminal_steps", "best_development_steps", "holdout_paired_comparisons")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
