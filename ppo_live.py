"""Screen measurements -> the frozen geometry-v1 PPO policy.

This module never reads game memory or changes a simulated/game bar position.
The caller applies the returned binary action through the existing mouse wrapper.
"""
from collections import deque
import csv
import hashlib
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = "training/runs/20260927_scaling_10m/best.zip"


def valid_roi(roi):
    return isinstance(roi, dict) and all(isinstance(roi.get(k), (int, float)) for k in
                                       ("left", "top", "width", "height")) and roi["width"] >= 2 and roi["height"] >= 30


def detect_progress(frame):
    """Read the bottom-anchored colored fill inside a manually selected full slot.

    A missing fill is unknown, not zero. Selection must exclude the wooden border.
    Returns (fraction or None, confidence). No simulated progress is substituted.
    """
    if frame is None or frame.shape[0] < 20 or frame.shape[1] < 2:
        return None, 0.0
    height, width = frame.shape[:2]
    inset = int(width * 0.25)
    hsv = cv2.cvtColor(frame[:, inset:max(inset + 1, width-inset)], cv2.COLOR_BGR2HSV)
    hue, sat, val = cv2.split(hsv)
    # The unfilled slot is brown (observed H=10, S=175, V=118), inside the
    # fill's hue range. Brightness and saturation must also distinguish it:
    # the actual red/yellow/green fill has V=255 and S=255 in saved captures.
    colored = (((hue <= 85) | (hue >= 170)) & (sat >= 150) & (val >= 200))
    occupied = (colored.mean(axis=1) >= 0.45).astype(np.uint8).reshape(-1, 1)
    occupied = cv2.morphologyEx(occupied, cv2.MORPH_CLOSE, np.ones((max(3, int(height * 0.008)), 1), np.uint8))[:, 0]
    edges = np.diff(np.r_[0, occupied, 0].astype(np.int16))
    starts, stops = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    candidates = [(int(a), int(b)) for a, b in zip(starts, stops)
                  if b >= height - max(3, int(height * 0.04)) and b-a >= max(2, int(height * 0.008))]
    if not candidates:
        return None, 0.0
    top, bottom = max(candidates, key=lambda pair: pair[1] - pair[0])
    confidence = float(colored[top:bottom].mean())
    return float(np.clip((height-top) / height, 0, 1)), confidence


class PPOController:
    decision_hz = 30

    def __init__(self, cfg, log=lambda message: None, log_path=None):
        # Lazy imports keep calibration and the legacy controller lightweight.
        training = str(ROOT / "training")
        if training not in sys.path:
            sys.path.insert(0, training)
        import torch
        from stable_baselines3 import PPO
        from fishing_sim.core import PhysicsConfig, LANE_TOP, LANE_HEIGHT
        from fishing_sim.observations import ObservationEncoder, OBS_SIZE, FRAME_SIZE, HISTORY, SCHEMA

        self.cfg, self.log = cfg, log
        self.Config, self.Encoder = PhysicsConfig, ObservationEncoder
        self.lane_top, self.lane_height = LANE_TOP, LANE_HEIGHT
        self.frame_size, self.history_size = FRAME_SIZE, HISTORY
        self.track_height = float(cfg["roi"]["height"])
        if not valid_roi(cfg.get("progress_roi")):
            raise ValueError("PPO 需要右侧捕获进度槽，请先点击“选择进度槽”。")
        path = Path(cfg.get("ppo_model", DEFAULT_MODEL))
        self.path = path if path.is_absolute() else ROOT / path
        if not self.path.is_file():
            raise FileNotFoundError(f"找不到 PPO 模型：{self.path}")
        manifest_path = self.path.parent / "manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("schema") != SCHEMA:
                raise ValueError("模型观测接口与 geometry-v1 不一致。")
        torch.set_num_threads(1)
        self.model = PPO.load(self.path, device="cpu")
        if self.model.observation_space.shape != (OBS_SIZE,) or self.model.action_space.n != 2:
            raise ValueError("需要 294 维观测、按住/松开两个动作的 PPO 模型。")
        self.level_setting = cfg.get("ppo_level", "auto")
        self.tackle = cfg.get("ppo_tackle", "none")
        self.level = 5 if self.level_setting == "auto" else int(self.level_setting)
        self.Config(level=self.level, tackle=self.tackle)
        self.episode = 0
        self.trace = None
        self.csv_writer = None
        self.trace_rows = 0
        if log_path:
            self.trace_path = Path(log_path).with_suffix(".ppo.csv")
            self.trace = self.trace_path.open("w", newline="", encoding="utf-8")
            self.csv_writer = csv.DictWriter(self.trace, fieldnames=[
                "episode", "elapsed", "frame_dt", "bar_top_px", "bar_bottom_px", "fish_y_px",
                "progress", "progress_confidence", "level", "tackle", "valid", "reason",
                "requested_action", "applied_action", "inference_ms", "vision_ms", "loop_to_input_ms",
            ])
            self.csv_writer.writeheader()
        self.log("[PPO] model=%s steps=%d sha256=%s" %
                 (self.path, self.model.num_timesteps, hashlib.sha256(self.path.read_bytes()).hexdigest()))
        self.log("[PPO] vision=20260927-faded-bar-bright-progress")
        self.log("[PPO] CPU / 30 Hz / geometry-v1 / observed progress / level=%s / tackle=%s" %
                 (self.level_setting, self.tackle))
        self.reset()

    def reset(self):
        self.encoder = None
        self.started = None
        self.last_valid = None
        self.commanded = False
        self.last_edge = None
        self.last_vel = 0.0
        self.last_pred = 0.0
        self.last = False
        self.status = "PPO 等待完整观测"
        self.heights = deque(maxlen=7)
        self.decision_times = deque(maxlen=30)
        self.record = None
        self.snapshot_count = 0
        self.gap_snapshot_count = 0
        self.last_gap_snapshot = None
        self.observation_time = None

    def _new_encoder(self):
        self.encoder = self.Encoder(self.Config(level=self.level, tackle=self.tackle))

    def update(self, now, bar, fish, progress=(None, 0.0), vision_ms=0.0):
        self.observation_time = now
        value, confidence = progress
        elapsed = max(0.0, now-self.started) if self.started is not None else 0.0
        dt = now-self.last_valid if self.last_valid is not None else 0.0
        reason = "" if bar is not None and fish is not None and value is not None else "缺少" + "/".join(
            name for name, item in (("绿条", bar), ("鱼", fish), ("进度", value)) if item is None)
        if bar is not None and not 0.10 <= (bar["bottom"]-bar["top"]) / self.track_height <= 0.55:
            reason = "绿条长度异常，请检查轨道选区"
        if elapsed >= 120:
            reason = "已超出模型训练的 120 秒时长"
        self.record = dict(episode=self.episode, elapsed=round(elapsed, 5), frame_dt=round(dt, 5),
                           bar_top_px=bar["top"] if bar else None, bar_bottom_px=bar["bottom"] if bar else None,
                           fish_y_px=fish["center"] if fish else None, progress=value,
                           progress_confidence=round(confidence, 3), level=self.level, tackle=self.tackle,
                           valid=not bool(reason), reason=reason, requested_action=0, applied_action=0,
                           inference_ms=0.0, vision_ms=round(vision_ms, 3), loop_to_input_ms=0.0)
        if reason:
            # Up to three missing frames retain the last action. Longer gaps release.
            want = self.commanded if self.last_valid is not None and dt < 0.10 and elapsed < 120 else False
            self.record["requested_action"] = int(want)
            self.status = "PPO：%s；%s" % (reason, "短暂保持" if want else "已松开")
            return want

        height = (bar["bottom"]-bar["top"]) / self.track_height * self.lane_height
        # Freeze the auto-estimate after seven frames; explicit UI level overrides it.
        if self.level_setting == "auto" and len(self.heights) < self.heights.maxlen:
            self.heights.append(height)
            estimated = int(np.clip(round((float(np.median(self.heights)) - 48 - (12 if self.tackle == "cork" else 0)) / 4), 0, 20))
            self.level = estimated
        if self.started is None:
            self.started = now
            self.last_edge = now
            self.episode += 1
            elapsed = 0.0
            self._new_encoder()
            self.log(f"[PPO] episode={self.episode}, measured_height={height:.2f}, level={self.level}")
        elif dt > 0.075:
            # No invented intermediate frames after missed/stalled captures.
            self._new_encoder()
        self.encoder.config_vector[0] = self.level / 20.0
        measurement = {
            "timestamp": elapsed,
            "fish_center": self.lane_top + fish["center"] / self.track_height * self.lane_height,
            "bar_top": self.lane_top + bar["top"] / self.track_height * self.lane_height,
            "bar_bottom": self.lane_top + bar["bottom"] / self.track_height * self.lane_height,
            "progress": float(value),
        }
        obs = self.encoder.encode(measurement, int(self.commanded), max(0.0, now-self.last_edge))
        started = time.perf_counter()
        action = int(self.model.predict(obs, deterministic=True)[0])
        inference_ms = (time.perf_counter()-started) * 1000
        self.last_valid = now
        self.last = bool(action)
        self.last_vel = float(self.encoder.history[-1, 4]) * self.track_height * 3
        self.last_pred = float(fish["center"])
        self.decision_times.append(now)
        hz = (len(self.decision_times)-1) / max(now-self.decision_times[0], 1e-6) if len(self.decision_times) > 1 else 0.0
        self.record.update(episode=self.episode, elapsed=round(elapsed, 5), level=self.level,
                           requested_action=action, inference_ms=round(inference_ms, 3))
        warning = " · 超出训练装备/等级" if self.tackle != "none" or self.level > 10 else ""
        self.status = f"PPO 控制中 · 进度 {value:.0%} · Lv.{self.level} · {hz:.1f} Hz · 推理 {inference_ms:.1f} ms{warning}"
        return bool(action)

    def input_applied(self, held, now):
        if held != self.commanded:
            self.last_edge = now
        self.commanded = bool(held)
        if self.record is not None and self.csv_writer is not None:
            self.record["applied_action"] = int(held)
            self.record["loop_to_input_ms"] = round(max(0.0, now - self.observation_time) * 1000, 3)
            self.csv_writer.writerow(self.record)
            self.trace_rows += 1
            if self.trace_rows % 30 == 0:
                self.trace.flush()

    def snapshot(self, frame, progress_frame, now):
        if self.trace is None or self.started is None:
            return
        missing = self.record is not None and not self.record["valid"]
        save_gap = (missing and self.gap_snapshot_count < 2 and
                    (self.last_gap_snapshot is None or now-self.last_gap_snapshot >= 0.25))
        save_regular = (not missing and self.snapshot_count < 2 and
                        (self.snapshot_count == 0 or now-self.started >= 1.0))
        if not save_gap and not save_regular:
            return
        stem = self.trace_path.with_suffix("")
        suffix = f"gap{self.gap_snapshot_count}" if save_gap else str(self.snapshot_count)
        tag = f"{stem.name}.episode{self.episode:03d}.{suffix}"
        cv2.imwrite(str(stem.with_name(tag + ".track.png")), frame)
        if progress_frame is not None:
            cv2.imwrite(str(stem.with_name(tag + ".progress.png")), progress_frame)
        if save_gap:
            self.gap_snapshot_count += 1
            self.last_gap_snapshot = now
        else:
            self.snapshot_count += 1

    def close(self):
        if self.trace is not None:
            self.trace.close()
            self.trace = None
