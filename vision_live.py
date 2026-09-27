"""Live screen-only adapter for the vision CNN and frozen geometry-v1 PPO.

No game memory or game state writes. All control is mouse down/up, while the
game owns the foreground. The GUI starts with control disabled.
"""
from collections import deque
import copy
import ctypes
from ctypes import wintypes
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import queue
import shutil
import subprocess
import sys
import threading
import time

import cv2
import mss
import numpy as np

ROOT = Path(__file__).resolve().parent
TRAINING = ROOT / "training"
if str(TRAINING) not in sys.path:
    sys.path.insert(0, str(TRAINING))

DEFAULTS = {
    "mode": "control", "level": "auto", "tackle": "none",
    # Real-game calibration from recorded progress/geometry, not Pufferdle's
    # renderer-only +5 mapping. See training/CATFISH_20260927_DIAGNOSIS_ZH.md.
    "fish_offset_native": 1.0,
    "calibration_profile": "real-game-20260927-v1",
    "vision_model": "training/vision_runs/20260927_v1_synthetic/best.pt",
    "ppo_model": "training/runs/20260927_scaling_10m/best.zip",
    "manual_panel": None,
    "full_auto": False,
    "auto_bite": "visual",
}
SETTINGS = ROOT / "vision_gui_config.json"


def load_settings():
    config = copy.deepcopy(DEFAULTS)
    if SETTINGS.exists():
        config.update(json.loads(SETTINGS.read_text(encoding="utf-8")))
    return config


def save_settings(config):
    temporary = SETTINGS.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(SETTINGS)


def absolute(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def enable_dpi_awareness():
    try:
        ctypes.windll.user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
    except (AttributeError, OSError):
        ctypes.windll.user32.SetProcessDPIAware()


class Windows:
    def __init__(self):
        self.user = ctypes.windll.user32
        self.kernel = ctypes.windll.kernel32
        self.user.GetForegroundWindow.restype = wintypes.HWND
        self.user.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
        self.user.IsWindow.argtypes = [wintypes.HWND]
        self.user.IsWindowVisible.argtypes = [wintypes.HWND]
        self.user.GetClientRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
        self.user.ClientToScreen.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.POINT)]
        self.kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        self.kernel.OpenProcess.restype = wintypes.HANDLE
        self.kernel.QueryFullProcessImageNameW.argtypes = [wintypes.HANDLE, wintypes.DWORD,
                                                         wintypes.LPWSTR, ctypes.POINTER(wintypes.DWORD)]
        self.kernel.CloseHandle.argtypes = [wintypes.HANDLE]

    def focused(self, hwnd):
        return bool(hwnd and self.user.GetForegroundWindow() == hwnd)

    def client(self, hwnd):
        if not hwnd or not self.user.IsWindow(hwnd):
            return None
        rect, origin = wintypes.RECT(), wintypes.POINT(0, 0)
        if not self.user.GetClientRect(hwnd, ctypes.byref(rect)):
            return None
        if not self.user.ClientToScreen(hwnd, ctypes.byref(origin)):
            return None
        w, h = rect.right - rect.left, rect.bottom - rect.top
        return {"left": origin.x, "top": origin.y, "width": w, "height": h} if w > 100 and h > 100 else None

    def find_game(self):
        matches = []
        callback_type = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

        @callback_type
        def callback(hwnd, unused):
            if not self.user.IsWindowVisible(hwnd):
                return True
            pid = wintypes.DWORD()
            self.user.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
            handle = self.kernel.OpenProcess(0x1000, False, pid.value)
            if not handle:
                return True
            try:
                buffer, count = ctypes.create_unicode_buffer(2048), wintypes.DWORD(2048)
                if self.kernel.QueryFullProcessImageNameW(handle, 0, buffer, ctypes.byref(count)):
                    name = Path(buffer.value).name.casefold()
                    if name in ("stardew valley.exe", "stardewmoddingapi.exe"):
                        rect = self.client(hwnd)
                        if rect:
                            matches.append((self.focused(hwnd), rect["width"] * rect["height"], hwnd))
            finally:
                self.kernel.CloseHandle(handle)
            return True

        self.user.EnumWindows(callback, 0)
        return max(matches)[2] if matches else None


class MouseLease:
    """Only releases buttons this process pressed. A watchdog limits stale holds."""
    def __init__(self, windows, active, shutdown, events):
        import pydirectinput
        self.api = pydirectinput
        self.api.PAUSE = 0
        self.api.FAILSAFE = False
        self.windows, self.active, self.shutdown = windows, active, shutdown
        self.events = events
        self.lock = threading.RLock()
        self.holding, self.hwnd, self.renewed = False, None, 0.
        self.lease_seconds = .18
        self.pressed_at = None
        self.release_deadline = None
        self.last_release = None
        self.watchdog = threading.Thread(target=self._watch, name="input-watchdog", daemon=True)
        self.watchdog.start()

    def set(self, down, hwnd=None, lease=.18):
        with self.lock:
            down = bool(down and self.active.is_set() and self.windows.focused(hwnd))
            self.renewed, self.hwnd = time.perf_counter(), hwnd
            self.lease_seconds = min(.40, max(.10, lease))
            if not down:
                self.release("controller")
            elif not self.holding:
                self.api.mouseDown()
                self.pressed_at = time.perf_counter()
                self.holding = True
                self.last_release = None
                self.events.put({"type": "input_edge", "time": self.pressed_at,
                                 "held": True, "reason": "controller"})
            return self.holding

    def hold_for(self, seconds, hwnd):
        """A bounded cast hold timed from mouseDown, independent of vision ticks."""
        with self.lock:
            self.release("start_timed_hold")
            if not self.set(True, hwnd):
                return None
            self.release_deadline = self.pressed_at + float(seconds)
            self.events.put({"type": "timed_hold", "time": self.pressed_at,
                             "seconds": float(seconds), "release_deadline": self.release_deadline})
            return self.pressed_at

    def move(self, x, y, hwnd):
        with self.lock:
            if self.active.is_set() and self.windows.focused(hwnd):
                self.windows.user.SetCursorPos(int(x), int(y))
                self.events.put({"type": "pointer_target", "time": time.perf_counter(),
                                 "x": int(x), "y": int(y)})

    def release(self, reason="release"):
        with self.lock:
            deadline = self.release_deadline
            self.release_deadline = None
            if self.holding:
                self.api.mouseUp()
                released_at = time.perf_counter()
                self.holding = False
                self.last_release = {"type": "input_edge", "time": released_at, "held": False,
                                     "reason": reason, "held_seconds": released_at-self.pressed_at}
                if deadline is not None:
                    self.last_release["target_seconds"] = deadline-self.pressed_at
                self.events.put(dict(self.last_release))

    def _watch(self):
        while not self.shutdown.is_set():
            delay, near_deadline = .02, False
            with self.lock:
                now = time.perf_counter()
                if self.holding:
                    if not self.active.is_set() or not self.windows.focused(self.hwnd):
                        self.release("focus_or_stop_watchdog")
                    elif self.release_deadline is not None:
                        remaining = self.release_deadline-now
                        if remaining <= 0:
                            self.release("cast_duration_complete")
                        else:
                            # Python's short sleep keeps the final release independent
                            # of the 30 Hz vision loop and its expensive captures.
                            near_deadline = remaining <= .025
                            delay = min(.001, remaining) if near_deadline else min(.02, remaining-.02)
                    elif now-self.renewed > self.lease_seconds:
                        self.release("stale_watchdog")
            if near_deadline:
                time.sleep(delay)
            else:
                self.shutdown.wait(delay)
        self.release("shutdown")


class FrozenPolicy:
    def __init__(self, path):
        from stable_baselines3 import PPO
        from fishing_sim.observations import OBS_SIZE, SCHEMA, ObservationEncoder
        from fishing_sim.core import PhysicsConfig
        path = absolute(path)
        manifest = json.loads((path.parent / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("schema") != SCHEMA:
            raise ValueError("PPO 观测接口不是 geometry-v1")
        self.model = PPO.load(path, device="cpu")
        if self.model.observation_space.shape != (OBS_SIZE,) or self.model.action_space.n != 2:
            raise ValueError("PPO 需要 294 维输入和两个按键动作")
        self.Encoder, self.Config = ObservationEncoder, PhysicsConfig
        self.reset(DEFAULTS)

    def reset(self, cfg):
        self.cfg = cfg
        self.level = 0 if cfg["level"] == "auto" else int(cfg["level"])
        self.heights = deque(maxlen=7)
        self.encoder = self.Encoder(self.Config(level=self.level, tackle=cfg["tackle"]))
        self.origin, self.last_time, self.edge = None, None, time.perf_counter()
        self.commanded = False

    def applied(self, pressed, timestamp):
        if bool(pressed) != self.commanded:
            self.edge = timestamp
        self.commanded = bool(pressed)

    def decide(self, geometry, captured):
        if self.origin is None:
            self.origin = captured
        if captured - self.origin > 120:
            raise TimeoutError("本局已达到策略训练的 120 秒上限")
        if self.last_time is not None and captured - self.last_time > .075:
            self.encoder = self.Encoder(self.Config(level=self.level, tackle=self.cfg["tackle"]))
        if self.cfg["level"] == "auto" and len(self.heights) < 7:
            self.heights.append(geometry["bar_bottom"] - geometry["bar_top"])
            cork = 12 if self.cfg["tackle"] == "cork" else 0
            self.level = int(np.clip(round((np.median(self.heights) - 48 - cork) / 4), 0, 20))
        self.encoder.config_vector[0] = self.level / 20.
        measurement = {**geometry, "timestamp": captured - self.origin}
        observation = self.encoder.encode(measurement, int(self.commanded), max(0., captured - self.edge))
        start = time.perf_counter()
        action = bool(int(self.model.predict(observation, deterministic=True)[0]))
        inference_ms = (time.perf_counter() - start) * 1000
        self.last_time = captured
        return action, inference_ms


class PanelMatcher:
    def __init__(self, assets):
        from fishing_vision.inference import PanelLocator
        self.coarse = PanelLocator(assets / "fishing_menu.png")
        self.menu, self.mask = self.coarse.menu, self.coarse.mask
        self.fish_sprite = cv2.imread(str(assets / "fish.png"), cv2.IMREAD_UNCHANGED)
        if self.fish_sprite is None or self.fish_sprite.shape[2] != 4:
            raise FileNotFoundError(assets / "fish.png")
        self.fish_sizes = {}
        self.reference = self.menu[self.mask > 0].astype(np.float32).ravel()
        self.reference_norm = np.linalg.norm(self.reference)
        rgba = cv2.imread(str(assets / "fishing_menu.png"), cv2.IMREAD_UNCHANGED)
        alpha = rgba[:, :, 3:4].astype(np.float32) / 255 if rgba.shape[2] == 4 else 1.
        self.context_reference = (rgba[:, :, :3] * alpha + 80 * (1-alpha)).astype(np.uint8)
        self.context_sizes = {}

    def inference_crop(self, crop, structure=None):
        """Make irrelevant scenery invariant after verifying the RAW wood rails.

        Keep native x=25..75 from the actual screenshot: both rails, fish lane,
        fish/bar/treasure and progress slot. Only the outer bubble/scenery/reel
        context is canonicalized. Negative/unconfirmed proposals remain raw.
        See training/SCENE_CHANGE_20260927_DIAGNOSIS_ZH.md.
        """
        if structure is None:
            structure = self.score(crop)
        if structure < .965:
            return crop
        h, w = crop.shape[:2]
        size = (w, h)
        if size not in self.context_sizes:
            # Only a handful of UI scales are used; don't retain arbitrary sizes.
            if len(self.context_sizes) >= 12:
                self.context_sizes.clear()
            self.context_sizes[size] = cv2.resize(self.context_reference, size,
                                                   interpolation=cv2.INTER_NEAREST)
        prepared = self.context_sizes[size].copy()
        left, right = round(25*w/94), round(75*w/94)
        prepared[:, left:right] = crop[:, left:right]
        return prepared

    def locate(self, frame):
        found = self.coarse.locate(frame, scales=(1., 1.25, 1.5, 1.75, 2., 2.25, 2.5, 3., 3.5, 4.))
        if found is None:
            return None
        pad = max(8, round(frame.shape[1] / 960) * 3)
        x0, y0 = max(0, found["left"] - pad), max(0, found["top"] - pad)
        x1, y1 = min(frame.shape[1], found["left"] + found["width"] + pad), min(frame.shape[0], found["top"] + found["height"] + pad)
        patch = frame[y0:y1, x0:x1]
        size = found["width"], found["height"]
        template = cv2.resize(self.menu, size, interpolation=cv2.INTER_NEAREST)
        mask = cv2.resize(self.mask, size, interpolation=cv2.INTER_NEAREST)
        if patch.shape[0] < size[1] or patch.shape[1] < size[0]:
            return None
        scores = cv2.matchTemplate(patch, template, cv2.TM_CCORR_NORMED, mask=mask)
        scores = np.nan_to_num(scores, nan=-1., posinf=-1., neginf=-1.)
        _, score, _, xy = cv2.minMaxLoc(scores)
        return {"left": x0 + xy[0], "top": y0 + xy[1], "width": size[0], "height": size[1],
                "structural_score": float(score)} if score >= .965 else None

    def score(self, crop):
        thumbnail = cv2.resize(crop, (94, 300), interpolation=cv2.INTER_AREA)
        values = thumbnail[self.mask > 0].astype(np.float32).ravel()
        return float(np.dot(values, self.reference) / max(np.linalg.norm(values) * self.reference_norm, 1e-8))

    def locate_near(self, patch, size):
        """Confirm an actor-relative panel guess within a few rendered pixels."""
        width, height = size
        if patch.shape[1] < width or patch.shape[0] < height:
            return None
        template = cv2.resize(self.menu, size, interpolation=cv2.INTER_NEAREST)
        mask = cv2.resize(self.mask, size, interpolation=cv2.INTER_NEAREST)
        scores = cv2.matchTemplate(patch, template, cv2.TM_CCORR_NORMED, mask=mask)
        scores = np.nan_to_num(scores, nan=-1., posinf=-1., neginf=-1.)
        _, score, _, xy = cv2.minMaxLoc(scores)
        return xy if score >= .965 else None

    def fish_sprite_center(self, crop):
        """Independent normal-fish location for CNN/treasure ambiguity.

        Only opaque sprite pixels participate, and x is restricted to the fish
        lane. Legendary icons are left to the CNN when this match is weak.
        """
        h, w = crop.shape[:2]
        key = (w, h)
        if key not in self.fish_sizes:
            if len(self.fish_sizes) >= 12:
                self.fish_sizes.clear()
            size = (max(4, round(19*w/94)), max(4, round(19*h/300)))
            sprite = cv2.resize(self.fish_sprite[:, :, :3], size, interpolation=cv2.INTER_NEAREST)
            alpha = cv2.resize(self.fish_sprite[:, :, 3], size, interpolation=cv2.INTER_NEAREST)
            self.fish_sizes[key] = (sprite, alpha)
        sprite, alpha = self.fish_sizes[key]
        x0 = max(0, round(27*w/94))
        x1 = min(w, round(40*w/94) + sprite.shape[1])
        if x1-x0 < sprite.shape[1] or h < sprite.shape[0]:
            return None, 0.
        scores = cv2.matchTemplate(crop[:, x0:x1], sprite, cv2.TM_CCORR_NORMED, mask=alpha)
        scores = np.nan_to_num(scores, nan=-1., posinf=-1., neginf=-1.)
        _, score, _, xy = cv2.minMaxLoc(scores)
        # The CNN's renderer label is sprite top + 10 native pixels.
        return (xy[1] + sprite.shape[0]/2) * 300/h + .5, float(score)


class FramePacer:
    """Absolute 30 Hz deadlines using Python's high-resolution Windows sleep.

    Event.wait uses a different Windows wait path and overshot short frame waits
    in the captured sessions. There is no backlog or synthetic catch-up frame.
    """
    def __init__(self, hz=30):
        self.period = 1 / hz
        self.reset()

    def reset(self):
        self.deadline = None

    def wait(self, active, shutdown):
        if self.deadline is None:
            self.deadline = time.perf_counter()
        while active.is_set() and not shutdown.is_set():
            now = time.perf_counter()
            remaining = self.deadline - now
            if remaining <= 0:
                if now - self.deadline >= self.period:
                    self.deadline = now
                self.deadline += self.period
                return True
            time.sleep(min(.004, remaining))
        self.reset()
        return False


class BackgroundPanelSearch:
    """One locator job at a time; results are proposals, never control inputs."""
    def __init__(self, assets, windows, shutdown):
        self.matcher = PanelMatcher(assets)
        self.windows, self.shutdown = windows, shutdown
        self.wake = threading.Event()
        self.lock = threading.Lock()
        self.job = self.result = None
        self.busy = False
        self.thread = threading.Thread(target=self._run, name="panel-locator", daemon=True)
        self.thread.start()

    def request(self, key, client, hwnd):
        with self.lock:
            if self.busy:
                return False
            self.job, self.busy = (key, dict(client), hwnd), True
            self.wake.set()
            return True

    def poll(self, key):
        with self.lock:
            result, self.result = self.result, None
        if result and result["key"] == key and time.perf_counter() - result["captured"] < 1.5:
            return result
        return None

    def _run(self):
        with mss.mss() as sct:
            while not self.shutdown.is_set():
                if not self.wake.wait(.1):
                    continue
                with self.lock:
                    job, self.job = self.job, None
                    self.wake.clear()
                if job is None:
                    continue
                key, client, hwnd = job
                start = time.perf_counter()
                captured = start
                proposal, error = None, None
                try:
                    if self.windows.focused(hwnd):
                        frame, captured = VisionSession._capture(sct, client)
                        proposal = self.matcher.locate(frame)
                except Exception as exc:
                    error = str(exc)
                with self.lock:
                    self.result = {"key": key, "roi": proposal, "captured": captured,
                                   "locator_ms": (time.perf_counter()-start)*1000, "error": error}
                    self.busy = False


def decode_geometry(prediction, crop, structure, offset, heights,
                    sprite_center=None, sprite_score=0., last_fish_center=None, require_fish=True):
    """Validate CNN geometry with its semantic mask; no HSV fallback.

    Low bar presence scores are shown/logged, but independent spatial evidence
    is required instead of merely reducing the presence threshold.
    """
    h, w = crop.shape[:2]
    scale, tx, ty = prediction["transform"]
    native = {key: float(value * 300 / h) for key, value in prediction["rows"].items()}
    scores = prediction["presence_scores"]
    sprite_valid = sprite_center is not None and sprite_score >= .94 and 5 <= sprite_center <= 285
    use_sprite = sprite_valid and (abs(sprite_center-native["fish_visual_center"]) >= 2
                                   or scores["fish"] < .7)
    fish_center = sprite_center if use_sprite else native["fish_visual_center"]
    geometry = {"fish_center": fish_center + offset,
                "bar_top": native["bar_top"], "bar_bottom": native["bar_bottom"],
                "progress": float(np.clip((292 - native["progress_top"]) / 288, 0, 1))}
    mask = prediction["mask"]
    nx = np.broadcast_to(((np.arange(mask.shape[1], dtype=np.float32) - tx) / scale * 94 / w)[None], mask.shape)
    ny = np.broadcast_to(((np.arange(mask.shape[0], dtype=np.float32) - ty) / scale * 300 / h)[:, None], mask.shape)
    legendary = mask == 2
    fish = (mask == 1) | legendary
    bar, progress, treasure = mask == 3, mask == 4, mask == 5
    units = (94 / w / scale) * (300 / h / scale)
    treasure_y = float(np.median(ny[treasure])) if treasure.any() else None
    treasure_near_fish = (treasure_y is not None and treasure.sum()*units >= 15 and
                          (abs(treasure_y - native["fish_visual_center"]) <= 24 or
                           (last_fish_center is not None and abs(treasure_y - last_fish_center) <= 24)))
    evidence = {"fish_area_native": float(fish.sum() * units), "bar_area_native": float(bar.sum() * units),
                "progress_area_native": float(progress.sum() * units), "structural_score": structure,
                "legendary_area_native": float(legendary.sum() * units),
                "treasure_area_native": float(treasure.sum() * units),
                "treasure_center_native": treasure_y, "treasure_near_fish": bool(treasure_near_fish),
                "fish_template_score": sprite_score,
                "fish_source": "sprite" if use_sprite else "cnn"}
    reason = ""
    length = geometry["bar_bottom"] - geometry["bar_top"]
    if structure < .965 or scores["panel"] < .85:
        reason = "未确认完整钓鱼面板"
    elif require_fish and treasure_near_fish and not sprite_valid and evidence["legendary_area_native"] < 15:
        reason = "宝箱遮挡鱼图标"
    elif require_fish and not sprite_valid and (scores["fish"] < .7 or evidence["fish_area_native"] < 15):
        reason = "鱼图标不可见或分割不足"
    elif require_fish and not sprite_valid and np.mean((nx[fish] >= 28) & (nx[fish] <= 56)) < .85:
        reason = "鱼图标不在钓鱼轨道"
    elif require_fish and not 5 <= fish_center <= 285:
        reason = "鱼坐标异常或处于开关动画"
    elif require_fish and not sprite_valid and abs(float(np.median(ny[fish])) - native["fish_visual_center"]) > 7:
        reason = "鱼热图与分割位置不一致"
    elif not (32 <= length <= 151 and 2 <= geometry["bar_top"] and geometry["bar_bottom"] <= 292):
        reason = "绿条边界异常或处于开关动画"
    elif evidence["bar_area_native"] < 40:
        reason = "绿条分割不足"
    elif np.mean((nx[bar] >= 30) & (nx[bar] <= 55)
                 & (ny[bar] >= geometry["bar_top"] - 4) & (ny[bar] <= geometry["bar_bottom"] + 4)) < .82:
        reason = "绿条热图与分割不一致"
    elif len(heights) >= 3 and abs(length - float(np.median(heights))) > 8:
        reason = "条长突变，可能被遮挡"
    elif evidence["progress_area_native"] < 4 or scores["progress"] < .6:
        reason = "捕获进度暂不可读"
    elif np.mean((nx[progress] >= 60) & (nx[progress] <= 74)) < .85:
        reason = "进度填充不在右侧槽内"
    elif not 2 <= native["progress_top"] <= 293:
        reason = "进度边界异常"
    elif abs(float(ny[progress].max()) - 291) > 5:
        reason = "进度槽底端不完整"
    elif abs(float(ny[progress].min()) - native["progress_top"]) > 5:
        reason = "进度热图与分割不一致"
    return geometry, native, evidence, reason


class VisionSession:
    def __init__(self, config):
        self.config = copy.deepcopy(config)
        self.windows = Windows()
        self.active, self.shutdown = threading.Event(), threading.Event()
        self.ready = threading.Event()
        self.messages = queue.SimpleQueue()
        self.input_events = queue.SimpleQueue()
        (ROOT / "logs").mkdir(exist_ok=True)
        self.app_log = ROOT / "logs" / ("vision_gui_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ".log")
        self.app_log_lock = threading.Lock()
        self.state_lock = threading.Lock()
        self.state = {"status": "正在加载视觉模型和 PPO…", "running": False, "ready": False}
        self.mouse = MouseLease(self.windows, self.active, self.shutdown, self.input_events)
        self.diagnostic_request = threading.Event()
        self.saving = threading.Event()
        self.ring = deque(maxlen=90)
        self.bite_ring = deque(maxlen=120)
        self.ring_lock = threading.Lock()
        self.preview = None
        self.generation = 0
        self.log_file = None
        self.path = None
        self.events_path = None
        self.policy = None
        self.autocycle = None
        self.worker = threading.Thread(target=self._worker, name="vision-ppo", daemon=True)
        self.worker.start()

    def publish(self, **values):
        with self.state_lock:
            self.state.update(values)

    def snapshot(self):
        with self.state_lock:
            return dict(self.state), self.preview

    def note(self, message):
        self.messages.put(message)
        with self.app_log_lock:
            with self.app_log.open("a", encoding="utf-8") as stream:
                stream.write(datetime.now().isoformat() + " " + message + "\n")

    def start(self, config):
        if not self.ready.is_set():
            return
        self.mouse.release("start_session")
        with self.state_lock:
            self.config = copy.deepcopy(config)
            self.generation += 1
        self.active.set()
        self.publish(running=True, status="正在启动 F1 全自动循环" if config.get("full_auto")
                     else "等待切回游戏；请手动抛竿和上钩")

    def stop(self, reason="已停止 · 输入已释放"):
        self.active.clear()
        self.mouse.release("user_stop")
        self.publish(running=False, status=reason)

    def close(self):
        self.stop()
        self.shutdown.set()

    def request_diagnostic(self):
        if self.saving.is_set():
            self.note("诊断片段仍在保存，请稍候。")
        else:
            self.diagnostic_request.set()

    def _open_log(self, cfg):
        self._close_log()
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self.path = ROOT / "logs" / ("vision_" + stamp)
        self.path.parent.mkdir(exist_ok=True)
        self.events_path = self.path.with_suffix(".jsonl")
        self.log_file = self.events_path.open("w", encoding="utf-8")
        self._record({"type": "session", "wall_time": datetime.now().isoformat(), "config": cfg,
                      "models": self.model_info, "input_contract": "Pufferdle native 94x300; lane y=6..288; progress y=4..292",
                      "fish_transform": "native visible center + fish_offset_native; real-game alignment is adjustable"})
        self.note("本次日志：" + str(self.events_path))

    def _record(self, record):
        if self.log_file:
            self.log_file.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")

    def _close_log(self):
        self._drain_inputs()
        if self.log_file:
            self.log_file.close()
            self.log_file = None

    def _close_auto(self):
        if self.autocycle:
            self.autocycle.close()
            self.autocycle = None

    def _drain_inputs(self):
        while not self.input_events.empty():
            self._record(self.input_events.get())

    def _save_diagnostic(self):
        self.diagnostic_request.clear()
        with self.ring_lock:
            samples = list(self.ring)
            bite_samples = list(self.bite_ring)
        if not samples and not bite_samples:
            self.note("尚未采集到画面；等待咬钩或小游戏出现后按 F9 保存。")
            return
        self.saving.set()
        folder = ROOT / "logs" / ("vision_clip_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
        config = copy.deepcopy(self.config)

        def write():
            try:
                folder.mkdir(parents=True)
                (folder / "frames.json").write_text(json.dumps({"model_info": self.model_info,
                    "config": config, "fps": 30, "timing": "Use capture timestamps; video FPS is only playback rate",
                    "frames": [record for _, record in samples],
                    "bite_frames": [record for _, record in bite_samples]}, ensure_ascii=False, indent=2), encoding="utf-8")
                ffmpeg = shutil.which("ffmpeg")
                for name, group in (("panel", samples), ("bite", bite_samples)):
                    if not group:
                        continue
                    frames = [frame for frame, _ in group]
                    # Dimension changes retain exact pixels as individual PNGs.
                    same_shape = len({frame.shape for frame in frames}) == 1
                    if ffmpeg and same_shape:
                        h, w = frames[0].shape[:2]
                        command = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo",
                                   "-pixel_format", "bgr24", "-video_size", f"{w}x{h}", "-framerate", "30",
                                   "-i", "pipe:0", "-an", "-c:v", "ffv1", str(folder / f"{name}.mkv")]
                        with (folder / f"{name}_ffmpeg.log").open("wb") as errors:
                            process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                                                       stderr=errors, creationflags=subprocess.CREATE_NO_WINDOW)
                            try:
                                for frame in frames:
                                    process.stdin.write(np.ascontiguousarray(frame).tobytes())
                                process.stdin.close()
                                if process.wait(timeout=45):
                                    raise RuntimeError("FFmpeg 编码失败，详见编码日志")
                            except BaseException:
                                process.kill()
                                process.wait()
                                raise
                    else:
                        for i, frame in enumerate(frames):
                            if not cv2.imwrite(str(folder / f"{name}_{i:04d}.png"), frame):
                                raise OSError("无法保存诊断帧")
                self.note(f"已保存面板 {len(samples)} 帧、咬钩区域 {len(bite_samples)} 帧与同步日志：{folder}")
            except Exception as exc:
                self.note(f"保存诊断失败：{exc}")
            finally:
                self.saving.clear()

        # Let an already requested save finish even after closing the GUI.
        threading.Thread(target=write, name="save-diagnostic", daemon=False).start()

    @staticmethod
    def _capture(sct, rect):
        desktop = sct.monitors[0]
        if (rect["left"] < desktop["left"] or rect["top"] < desktop["top"]
                or rect["left"] + rect["width"] > desktop["left"] + desktop["width"]
                or rect["top"] + rect["height"] > desktop["top"] + desktop["height"]):
            raise ValueError("游戏/面板未完整显示在屏幕内")
        t0 = time.perf_counter()
        frame = np.asarray(sct.grab(rect))[:, :, :3].copy()
        return frame, (t0 + time.perf_counter()) / 2

    def _worker(self):
        import traceback
        try:
            import torch
            from fishing_vision.inference import VisionPredictor
            from auto_fishing import GaugeReader, CAST_HOLD_SECONDS
            from audio_bite import prepare_audio_backend
            cv2.setNumThreads(1)
            torch.set_num_threads(2)
            if not torch.cuda.is_available():
                raise RuntimeError("未检测到 CUDA，请使用“启动视觉PPO助手.bat”的独立运行环境")
            self.vision = VisionPredictor(absolute(self.config["vision_model"]), "cuda")
            self.policy = FrozenPolicy(self.config["ppo_model"])
            self.matcher = PanelMatcher(TRAINING / "vision_data" / "assets")
            self.gauge_reader = GaugeReader(ROOT / "auto_assets")
            # Pay CUDA kernel/allocation startup costs before enabling F1.
            for _ in range(3):
                self.vision.predict([self.matcher.inference_crop(self.matcher.menu)])
            prepare_audio_backend(self.note)
            self.panel_search = BackgroundPanelSearch(TRAINING / "vision_data" / "assets",
                                                      self.windows, self.shutdown)
            self.model_info = {"vision_step": self.vision.step, "ppo_steps": int(self.policy.model.num_timesteps),
                               "gpu": torch.cuda.get_device_name(), "torch": str(torch.__version__),
                               "vision_preprocessing": "raw-rails-gated-canonical-outer-context-v1",
                               "runtime_adapter": "f1-start-layout-energy-v3",
                               "vision_sha256": hashlib.sha256(absolute(self.config["vision_model"]).read_bytes()).hexdigest(),
                               "ppo_sha256": hashlib.sha256(absolute(self.config["ppo_model"]).read_bytes()).hexdigest()}
            self.ready.set()
            (ROOT / "logs" / "vision_gui_ready.json").write_text(json.dumps({"pid": os.getpid(),
                "loaded_at": datetime.now().isoformat(), "initial_state": "stopped",
                "features": ["f1_auto_cycle_v2", "fixed_cast_1050ms", "deadline_pacing_30hz",
                             "async_panel_locator", "actor_bite_confirmation", "per_thread_audio_com",
                             "real_game_geometry_calibration_v1", "canonical_outer_context_v1",
                             "panel_presence_episode_tracking"],
                "runtime_refinements": ["actor_panel_prior_v1", "day_night_bite_delta_v1",
                                        "normal_fish_sprite_crosscheck_v1", "treasure_occlusion_hold_120ms",
                                        "start_panel_layout_guard_v1", "validated_energy_confirmation_v1"],
                "cast_hold_seconds": CAST_HOLD_SECONDS,
                "geometry_calibration": {"fish_offset_native": self.config["fish_offset_native"],
                                         "level": self.config["level"]},
                "models": self.model_info},
                ensure_ascii=False, indent=2), encoding="utf-8")
            self.publish(ready=True, status="模型就绪 · 游戏中 F1 全自动；或点击小游戏辅助", models=self.model_info)
            self.note(f"视觉第 {self.vision.step:,} 步 + PPO {self.policy.model.num_timesteps:,} 步；{torch.cuda.get_device_name()}")
            with mss.mss() as sct:
                self._loop(sct)
        except Exception as exc:
            self.ready.clear()
            self.stop("运行错误：" + str(exc))
            error = traceback.format_exc()
            self.note(error)
            logs = ROOT / "logs"
            logs.mkdir(exist_ok=True)
            (logs / ("vision_error_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".log")).write_text(error, encoding="utf-8")
            self.publish(ready=False, error=str(exc))
        finally:
            self.active.clear()
            self.shutdown.set()
            self.mouse.release()
            self._close_auto()
            self._close_log()

    def _loop(self, sct):
        generation, hwnd, roi = -1, None, None
        last_find, next_search, last_good, last_preview = 0., 0., 0., 0.
        previous_client = None
        confirmed, inside, episode, count = 0, False, 0, 0
        heights, periods = deque(maxlen=15), deque(maxlen=30)
        pacer = FramePacer(30)
        panel_hint, next_hint, previous_capture = None, 0., None
        last_panel_seen = 0.
        last_fish = None
        while not self.shutdown.is_set():
            self._drain_inputs()
            if self.diagnostic_request.is_set():
                self._save_diagnostic()
            if not self.active.is_set():
                pacer.reset()
                self.mouse.release()
                self._close_auto()
                self._close_log()
                self.shutdown.wait(.035)
                continue
            if not pacer.wait(self.active, self.shutdown):
                continue
            t0 = time.perf_counter()
            locator_ms, roi_source = 0., "tracked"
            if generation != self.generation:
                self._close_auto()
                cfg = copy.deepcopy(self.config)
                generation = self.generation
                self._open_log(cfg)
                self.policy.reset(cfg)
                if cfg.get("full_auto"):
                    from auto_fishing import AutoFishingCycle
                    self.autocycle = AutoFishingCycle(self, cfg)
                self.publish(auto_phase="arming" if self.autocycle else None, auto_cast=0,
                             auto_power=None, auto_energy=None, auto_hold_elapsed=0., auto_hold_seconds=None)
                roi, previous_client = None, None
                last_find = 0.
                next_search, last_good, confirmed, inside, episode, count = 0., 0., 0, False, 0, 0
                heights.clear(); periods.clear()
                previous_capture = None
                last_panel_seen = 0.
                last_fish = None
                self.publish(hz=0., bite_hz=0., bite_visual_frames=0, bite_audio=None)
                with self.ring_lock:
                    self.ring.clear()
                    self.bite_ring.clear()
            if hwnd is None or not self.windows.user.IsWindow(hwnd):
                if t0 - last_find >= .7:
                    hwnd = self.windows.find_game()
                    last_find = t0
            if not self.windows.focused(hwnd):
                self.mouse.release()
                if self.autocycle:
                    self.stop("全自动已停止：游戏失去焦点；回到水边后按 F1 重新开始")
                    continue
                if inside or confirmed:
                    self._record({"type": "focus_lost", "time": t0, "episode": episode})
                    self.policy.reset(cfg)
                inside, confirmed, roi = False, 0, None
                last_panel_seen = 0.
                last_fish = None
                heights.clear(); periods.clear()
                self.publish(status="请切回星露谷游戏窗口" if hwnd else "等待星露谷游戏打开", holding=False, focused=False)
                pacer.reset()
                self.shutdown.wait(.06)
                continue
            client = self.windows.client(hwnd)
            if client is None:
                self.mouse.release()
                hwnd = None
                continue
            if previous_client != client:
                self.mouse.release()
                if previous_client is not None and self.autocycle:
                    self.stop("全自动已停止：游戏窗口改变；请重新按 F1")
                    continue
                roi, confirmed, inside = None, 0, False
                heights.clear()
                self.policy.reset(cfg)
                previous_client = client
                panel_hint, next_hint, previous_capture = None, 0., None
                last_panel_seen = 0.
                last_fish = None
                periods.clear()
            if self.autocycle:
                self.autocycle.tick(sct, client, hwnd)
                if not self.active.is_set():
                    continue
                if not self.autocycle.should_scan():
                    self._drain_inputs()
                    continue
            manual = cfg.get("manual_panel")
            if roi is None:
                self.mouse.release()
                if manual:
                    if manual["client_size"] != [client["width"], client["height"]]:
                        self.publish(status="窗口尺寸已改变，请重新框选或使用自动定位", holding=False)
                        continue
                    roi = dict(manual["roi"])
                    roi_source = "manual"
                else:
                    if not self.autocycle:
                        self.publish(status="寻找钓鱼面板 · 请手动抛竿和上钩", focused=True, holding=False)
                    # The casting gauge gives an actor-relative panel position.
                    # A small masked rail search is faster than the first global
                    # multi-scale scan; geometry still validates the CNN result.
                    prior = self.autocycle.panel_prior(client) if self.autocycle else None
                    if prior:
                        pad = max(4, round(prior["height"] / 75))
                        x0, y0 = max(0, prior["left"]-pad), max(0, prior["top"]-pad)
                        x1 = min(client["width"], prior["left"]+prior["width"]+pad)
                        y1 = min(client["height"], prior["top"]+prior["height"]+pad)
                        search = {"left": client["left"]+x0, "top": client["top"]+y0,
                                  "width": x1-x0, "height": y1-y0}
                        patch, _ = self._capture(sct, search)
                        found = self.matcher.locate_near(patch, (prior["width"], prior["height"]))
                        if found is not None:
                            roi = {"left": x0+found[0], "top": y0+found[1],
                                   "width": prior["width"], "height": prior["height"]}
                            roi_source = "actor_panel_prior"
                    search_key = (generation, hwnd, *[client[k] for k in ("left", "top", "width", "height")])
                    result = self.panel_search.poll(search_key) if roi is None else None
                    if result is not None:
                        roi, locator_ms = result["roi"], result["locator_ms"]
                        roi_source = "background_locator"
                        if result["error"]:
                            self._record({"type": "locator_error", "time": t0, "error": result["error"]})
                    # Reuse only a location; every frame and every CNN estimate is
                    # freshly captured. This avoids another global search per fish.
                    if roi is None and panel_hint is not None and t0 >= next_hint:
                        next_hint = t0 + .065
                        hint_rect = {**panel_hint, "left": client["left"]+panel_hint["left"],
                                     "top": client["top"]+panel_hint["top"]}
                        hint_crop, _ = self._capture(sct, hint_rect)
                        if self.matcher.score(hint_crop) >= .965:
                            roi, roi_source = dict(panel_hint), "previous_panel_location"
                    if roi is None:
                        if t0 >= next_search and self.panel_search.request(search_key, client, hwnd):
                            next_search = t0 + (.65 if panel_hint else .18)
                        continue
                if not (roi["left"] >= 0 and roi["top"] >= 0
                        and roi["left"] + roi["width"] <= client["width"]
                        and roi["top"] + roi["height"] <= client["height"]):
                    roi = None
                    self.publish(status="面板选区越界，请重新定位")
                    continue
                last_good = time.perf_counter()
                previous_capture = None
                periods.clear()
                self.note(f"面板候选：{roi['width']}×{roi['height']}，窗口内 ({roi['left']}, {roi['top']})")
            screen_roi = {"left": client["left"] + roi["left"], "top": client["top"] + roi["top"],
                          "width": roi["width"], "height": roi["height"]}
            capture_start = time.perf_counter()
            crop, captured = self._capture(sct, screen_roi)
            capture_end = time.perf_counter()
            # A generated outer context must never validate its own structure.
            # The raw screenshot is also retained in previews and diagnostics.
            structure = self.matcher.score(crop)
            prepared = self.matcher.inference_crop(crop, structure)
            context_end = time.perf_counter()
            predicted = self.vision.predict([prepared])[0]
            inference_end = time.perf_counter()
            sprite_center, sprite_score = self.matcher.fish_sprite_center(crop)
            geometry, native, evidence, reason = decode_geometry(predicted, crop, structure,
                                                  float(cfg["fish_offset_native"]), heights,
                                                  sprite_center, sprite_score,
                                                  last_fish[1] if last_fish and captured-last_fish[0] <= .15 else None)
            decode_end = time.perf_counter()
            vision_ms = (time.perf_counter() - t0) * 1000
            capture_ms = (capture_end - capture_start) * 1000
            context_ms = (context_end - capture_end) * 1000
            cnn_ms = (inference_end - context_end) * 1000
            geometry_ms = (decode_end - inference_end) * 1000
            frame_interval_ms = None if previous_capture is None else (captured - previous_capture) * 1000
            previous_capture = captured
            age_ms = (time.perf_counter() - captured) * 1000
            if age_ms > 80:
                reason = "观测已超时，释放输入"
            if not self.active.is_set() or not self.windows.focused(hwnd) or generation != self.generation:
                reason = "已停止或游戏失去焦点"
            panel_present = structure >= .965 and predicted["presence_scores"]["panel"] >= .80
            if panel_present:
                last_panel_seen = captured
            if self.autocycle:
                self.autocycle.panel_observed(panel_present, valid=not bool(reason))
            request, applied, ppo_ms, occlusion_hold = False, False, 0., False
            if not reason:
                last_good = captured
                last_fish = (captured, geometry["fish_center"] - float(cfg["fish_offset_native"]))
                heights.append(geometry["bar_bottom"] - geometry["bar_top"])
                confirmed += 1
                if confirmed >= (1 if self.autocycle else 2) and not inside:
                    inside = True
                    panel_hint = {key: roi[key] for key in ("left", "top", "width", "height")}
                    episode += 1
                    self.policy.reset(cfg)
                    if self.autocycle:
                        self.autocycle.minigame_started()
                    self._record({"type": "minigame_start", "time": captured, "episode": episode})
                    self.note(f"小游戏 #{episode}：观测通过，{'PPO 接管' if cfg['mode'] == 'control' else '只观察'}")
                if inside:
                    observed_button = (bool(self.windows.user.GetAsyncKeyState(0x01) & 0x8000)
                                       if cfg["mode"] == "observe" else self.mouse.holding)
                    self.policy.applied(observed_button, captured)
                    try:
                        request, ppo_ms = self.policy.decide(geometry, captured)
                    except TimeoutError as exc:
                        self.stop(str(exc))
                        continue
                    # Re-check age after PPO too; a slow frame never receives a new lease.
                    if time.perf_counter() - captured > .08 or generation != self.generation:
                        reason = "决策已超时，释放输入"
                        self.mouse.release()
                    else:
                        applied = self.mouse.set(request if cfg["mode"] == "control" else False, hwnd)
                    if cfg["mode"] == "control":
                        self.policy.applied(applied, time.perf_counter())
            else:
                confirmed = 0
                occlusion_hold = bool(reason == "宝箱遮挡鱼图标" and inside
                                      and captured-last_good <= .12 and cfg["mode"] == "control")
                if occlusion_hold:
                    applied = self.mouse.set(self.mouse.holding, hwnd)
                else:
                    self.mouse.release()
                if cfg["mode"] == "control":
                    self.policy.applied(applied, time.perf_counter())
            input_done = time.perf_counter()
            self._drain_inputs()
            now = time.perf_counter()
            # A visible panel is the same fight across brief geometry gaps.
            # Do not repeatedly reset episode identity/time origin.
            # Unconfirmed candidate ROIs must still time out and be re-searched.
            if now - last_good > .6 and (not inside or now - last_panel_seen > .6):
                if inside:
                    self._record({"type": "minigame_end_or_lost", "time": now, "episode": episode})
                    self.note(f"小游戏 #{episode}：面板退出或持续丢失；结果未自动判定")
                inside, confirmed = False, 0
                self.policy.reset(cfg)
                heights.clear()
                last_fish = None
                roi = None
            elif (not inside and roi_source in ("actor_panel_prior", "background_locator", "previous_panel_location")
                  and predicted["presence_scores"]["panel"] < .5):
                # A rail-like world texture is only a proposal. Re-scan on the
                # next frame instead of staying on it for the full timeout.
                roi = None
            periods.append(captured)
            hz = (len(periods) - 1) / max(periods[-1] - periods[0], 1e-6) if len(periods) > 1 else 0.
            if reason:
                status = reason + (" · 短暂保持上一输入" if occlusion_hold else " · 已松开")
            elif not inside:
                status = f"正在确认完整观测 {confirmed}/{1 if self.autocycle else 2}"
            else:
                status = ("PPO 控制中" if cfg["mode"] == "control" else "只观察 · 不发送输入")
                if predicted["presence_scores"]["bar"] < .7:
                    status += " · 绿条分数低，几何校验通过"
            if self.autocycle:
                status = f"全自动 · 第 {self.autocycle.casts} 杆 · " + status
            record = {"type": "frame", "captured": captured, "wall_time": time.time(), "episode": episode,
                      "screen_roi": screen_roi, "native_rows": native, "geometry": geometry,
                      "presence_scores": predicted["presence_scores"], "evidence": evidence,
                      "reason": reason, "active_minigame": inside, "requested_action": int(request),
                      "applied_action": int(applied), "occlusion_hold": occlusion_hold,
                      "mode": cfg["mode"], "level": self.policy.level,
                      "capture_ms": capture_ms, "cnn_ms": cnn_ms, "geometry_ms": geometry_ms,
                      "context_ms": context_ms, "canonical_context": prepared is not crop,
                      "panel_present": panel_present,
                      "frame_interval_ms": frame_interval_ms, "actual_hz": hz,
                      "locator_ms": locator_ms, "roi_source": roi_source,
                      "vision_ms": vision_ms, "ppo_ms": ppo_ms, "capture_to_input_ms": (input_done - captured) * 1000}
            self._record(record)
            with self.ring_lock:
                self.ring.append((crop, record))
            count += 1
            if count % 30 == 0 and self.log_file:
                self.log_file.flush()
            self.publish(status=status, focused=True, holding=applied, suggested=request, episode=episode,
                         hz=hz, vision_ms=vision_ms, ppo_ms=ppo_ms, age_ms=(input_done - captured) * 1000,
                         capture_ms=capture_ms, cnn_ms=cnn_ms, geometry_ms=geometry_ms,
                         scores=predicted["presence_scores"], geometry=geometry, native=native,
                         level=self.policy.level, roi=screen_roi, evidence=evidence, valid=not bool(reason))
            if now - last_preview >= .10:
                with self.state_lock:
                    self.preview = (crop, record)
                last_preview = now
