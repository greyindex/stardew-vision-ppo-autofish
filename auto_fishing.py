"""F1 fishing cycle using screenshots, optional loopback audio and mouse input.

F1 is the user's indication that they are stationary at water with a rod equipped.
Cast timing uses the user's measured 1.05-second hold for this game setup.
No navigation, game memory or item/menu manipulation.
"""
from pathlib import Path
from collections import deque
import time

import cv2
import numpy as np

CAST_HOLD_SECONDS = 1.05


def bounds(frame, rect):
    x, y, w, h = [int(rect[k]) for k in ("left", "top", "width", "height")]
    if min(x, y) < 0 or w < 2 or h < 2 or x + w > frame.shape[1] or y + h > frame.shape[0]:
        return None
    return frame[y:y+h, x:x+w]


def correlation(image, reference, mask):
    image = cv2.resize(image, (reference.shape[1], reference.shape[0]), interpolation=cv2.INTER_AREA)
    a, b = image[mask > 0].astype(np.float32).ravel(), reference[mask > 0].astype(np.float32).ravel()
    return float(np.dot(a, b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-8))


class GaugeReader:
    def __init__(self, assets):
        self.power = cv2.imread(str(assets / "power_meter.png"))
        self.energy = cv2.imread(str(assets / "energy_meter.png"))
        if self.power is None or self.energy is None:
            raise FileNotFoundError("缺少 auto_assets 中的蓄力条/精力条素材")
        self.power_mask = np.zeros(self.power.shape[:2], np.uint8)
        self.power_mask[3:9, 9:-9] = 255
        self.power_mask[-9:-3, 9:-9] = 255
        self.power_mask[10:-10, 2:8] = 255
        self.power_mask[10:-10, -8:-2] = 255
        self.energy_mask = np.zeros(self.energy.shape[:2], np.uint8)
        self.energy_mask[7:42, 10:36] = 255  # Includes the E label, not a generic green strip.
        self.energy_mask[46:220, 4:8] = 255
        self.energy_mask[46:220, 37:41] = 255
        self.energy_mask[218:223, 7:39] = 255

    @staticmethod
    def locate(image, template, mask, scales, threshold, max_width, validator=None):
        shrink = min(1., max_width / image.shape[1])
        small = cv2.resize(image, None, fx=shrink, fy=shrink, interpolation=cv2.INTER_AREA)
        candidates = []
        for scale in scales:
            w, h = round(template.shape[1] * scale * shrink), round(template.shape[0] * scale * shrink)
            if min(w, h) < 8 or w > small.shape[1] or h > small.shape[0]:
                continue
            reference = cv2.resize(template, (w, h), interpolation=cv2.INTER_AREA)
            selected = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            values = cv2.matchTemplate(small, reference, cv2.TM_CCORR_NORMED, mask=selected)
            values = np.nan_to_num(values, nan=-1., posinf=-1., neginf=-1.)
            for _ in range(3 if validator else 1):
                _, score, _, xy = cv2.minMaxLoc(values)
                if score < threshold:
                    break
                candidates.append({"left": round(xy[0] / shrink), "top": round(xy[1] / shrink),
                                   "width": round(template.shape[1] * scale), "height": round(template.shape[0] * scale),
                                   "score": float(score)})
                values[max(0, xy[1]-h//2):xy[1]+h//2+1, max(0, xy[0]-w//2):xy[0]+w//2+1] = -1
        # Undo the coarse downsampling's position rounding at native resolution.
        pad = max(6, round(3 / shrink))
        for best in sorted(candidates, key=lambda r: r["score"], reverse=True):
            x0, y0 = max(0, best["left"] - pad), max(0, best["top"] - pad)
            x1 = min(image.shape[1], best["left"] + best["width"] + pad)
            y1 = min(image.shape[0], best["top"] + best["height"] + pad)
            patch = image[y0:y1, x0:x1]
            size = best["width"], best["height"]
            if patch.shape[0] < size[1] or patch.shape[1] < size[0]:
                continue
            values = cv2.matchTemplate(patch, cv2.resize(template, size, interpolation=cv2.INTER_NEAREST),
                cv2.TM_CCORR_NORMED, mask=cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST))
            values = np.nan_to_num(values, nan=-1., posinf=-1., neginf=-1.)
            _, score, _, xy = cv2.minMaxLoc(values)
            best.update(left=x0+xy[0], top=y0+xy[1], score=float(score))
            if score >= threshold and (validator is None or validator(best)):
                return best
        return None

    def locate_power(self, frame, baseline):
        def is_new_meter(rect):
            current, previous = bounds(frame, rect), bounds(baseline, rect)
            return (current is not None and previous is not None
                    and np.abs(current.astype(np.float32)-previous).mean() >= 8
                    and self.power_fill(current) is not None)
        rect = self.locate(frame, self.power, self.power_mask,
                           (.5, .625, .75, .875, 1., 1.125, 1.25, 1.5, 2.), .965, 960, is_new_meter)
        if rect is None:
            return None
        current, previous = bounds(frame, rect), bounds(baseline, rect)
        if current is None or previous is None or np.abs(current.astype(float)-previous).mean() < 8:
            return None
        fill = self.power_fill(current)
        return (rect, fill) if fill is not None else None

    def power_fill(self, crop):
        if correlation(crop, self.power, self.power_mask) < .965:
            return None
        image = cv2.resize(crop, (185, 48), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(image[12:36, 10:175], cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        colored = (((h <= 88) | (h >= 170)) & (s >= 110) & (v >= 75))
        columns = (colored.mean(0) > .60).astype(np.uint8)
        columns = cv2.morphologyEx(columns[None], cv2.MORPH_CLOSE, np.ones((1, 3), np.uint8))[0]
        if columns[:4].mean() < .5:
            return None
        empty = np.flatnonzero(columns == 0)
        end = int(empty[0]) if len(empty) else len(columns)
        return float(end / len(columns))

    def energy_fraction(self, frame):
        x0, y0 = round(frame.shape[1] * .85), round(frame.shape[0] * .4)
        right = frame[y0:, x0:]
        def readable_energy(rect):
            crop = bounds(right, rect)
            return crop is not None and self.energy_crop_fraction(crop) is not None
        rect = self.locate(right, self.energy, self.energy_mask, (.5, .75, 1., 1.25, 1.5, 2.),
                           .975, 500, readable_energy)
        if rect is None:
            return None
        image = bounds(right, rect)
        if image is None:
            return None
        return self.energy_crop_fraction(image)

    def energy_crop_fraction(self, image):
        """Read only a verified E gauge; an unreadable fill is not zero energy."""
        image = cv2.resize(image, (46, 228), interpolation=cv2.INTER_AREA)
        label = cv2.cvtColor(image[7:42, 10:36], cv2.COLOR_BGR2GRAY).astype(np.float32).ravel()
        reference = cv2.cvtColor(self.energy[7:42, 10:36], cv2.COLOR_BGR2GRAY).astype(np.float32).ravel()
        label -= label.mean(); reference -= reference.mean()
        contrast = float(np.dot(label, reference) / max(np.linalg.norm(label)*np.linalg.norm(reference), 1e-8))
        if contrast < .80:
            return None
        hsv = cv2.cvtColor(image[52:216, 11:33], cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        label_value = cv2.cvtColor(image[7:42, 10:36], cv2.COLOR_BGR2HSV)[:, :, 2]
        min_value = max(40., float(np.percentile(label_value, 90)) * .65)
        filled = ((((h <= 85) | (h >= 170)) & (s >= 80) & (v >= min_value)).mean(1) > .45)
        if not filled.any():
            return None
        if not filled[-4:].any():
            return None
        # A filled meter is one bottom-anchored region, not scattered scenery.
        first = np.flatnonzero(filled)[0]
        if filled[first:].mean() < .90:
            return None
        return float((len(filled) - np.flatnonzero(filled)[0]) / len(filled))


class AutoFishingCycle:
    LABELS = {"arming": "准备抛竿", "charge": "蓄力", "settle": "等待鱼漂落水",
              "wait": "等待咬钩", "hook": "提竿", "hook_wait": "等待小游戏 / 直接收获",
              "playing": "PPO 钓鱼", "resolve": "等待收竿结束", "collect": "确认收获", "cooldown": "准备下一杆"}

    def __init__(self, session, config):
        self.session, self.config = session, config
        self.generation = session.generation
        self.reader = session.gauge_reader
        self.phase, self.entered = "arming", time.perf_counter()
        self.world = None
        self.power_rect = None
        self.actor_rect = None
        self.watch_rect = None
        self.idle = self.wait_pose = self.pose_mask = self.wait_watch = None
        self.pose = None
        self.last_pose_time = 0.
        self.idle_since = self.changed_since = None
        self.last_panel = self.last_valid = time.perf_counter()
        self.cast_started = None
        self.collect_count, self.casts = 0, 0
        self.energy = None
        self.visual_count = 0
        self.last_visual = 0.
        self.last_visual_x = None
        self.pending_audio = None
        self.last_bite_log = self.last_bite_preview = 0.
        self.bite_times = deque(maxlen=30)
        self.audio = None
        if config.get("auto_bite", "visual_audio") == "visual_audio":
            from audio_bite import AudioBiteDetector
            self.audio = AudioBiteDetector({"bite_sound_ratio": 3.5, "bite_sound_floor": .03,
                                          "bite_refractory": 1.0}, log=session.note)

    def close(self):
        if self.audio:
            self.audio.arm(False)
            self.audio.stop()
            self.audio = None

    def transition(self, phase, why):
        self.phase, self.entered = phase, time.perf_counter()
        if self.audio:
            self.audio.arm(phase in ("settle", "wait"))
        if phase == "charge":
            self.visual_count = 0
            self.last_visual = 0.
            self.last_visual_x = None
            self.pending_audio = None
            self.bite_times.clear()
            self.wait_pose = self.pose_mask = self.wait_watch = None
        self.session._record({"type": "auto_state", "time": self.entered, "state": phase,
                              "cast": self.casts, "reason": why})
        self.session.note(f"[全自动 {self.casts}] {self.LABELS[phase]}：{why}")

    def fail(self, message):
        if self.audio:
            self.audio.arm(False)
        self.session._record({"type": "auto_stop", "time": time.perf_counter(), "reason": message})
        self.session.note("全自动已停止：" + message)
        self.session.stop("全自动已停止：" + message)

    def low_energy_confirmed(self, sct, client):
        if self.energy is None or self.energy > .12:
            return False
        first = self.energy
        self.energy = self.reader.energy_fraction(self.capture(sct, client))
        self.session._record({"type": "energy_confirmation", "time": time.perf_counter(),
                              "first": first, "second": self.energy})
        return self.energy is not None and self.energy <= .12

    def should_scan(self):
        # Ordinary rods cannot open a minigame until after our hook click.
        # Avoid a continuous full-screen locator competing with bite detection.
        return (self.phase in ("hook_wait", "playing", "resolve")
                or (self.config.get("auto_bite") == "enchanted" and self.phase in ("settle", "wait")))

    def panel_prior(self, client):
        """Predict the minigame panel from the observed casting meter.

        Independent recordings at different player locations have the same
        meter-to-panel offset. A narrow rail match must still confirm the ROI.
        """
        if self.power_rect is None:
            return None
        scale = self.power_rect["height"] / 48.0
        roi = {"left": round(self.power_rect["left"] - 152 * scale),
               "top": round(self.power_rect["top"] - 146 * scale),
               "width": round(188 * scale), "height": round(600 * scale)}
        if (roi["left"] < 0 or roi["top"] < 0 or roi["left"] + roi["width"] > client["width"]
                or roi["top"] + roi["height"] > client["height"]):
            return None
        return roi

    def panel_observed(self, present, valid=False):
        now = time.perf_counter()
        if present:
            self.last_panel = now
        if valid:
            self.last_valid = now

    def minigame_started(self):
        self.last_panel = self.last_valid = time.perf_counter()
        if self.phase != "playing":
            self.session.mouse.release("auto_to_ppo")
            self.transition("playing", "完整小游戏观测已确认")

    def capture(self, sct, client, rect=None):
        capture_rect = dict(client) if rect is None else {
            "left": client["left"] + rect["left"], "top": client["top"] + rect["top"],
            "width": rect["width"], "height": rect["height"]}
        image, self.last_capture = self.session._capture(sct, capture_rect)
        return image

    def calibrate_actor(self, rect):
        x, y, w, h = [rect[k] for k in ("left", "top", "width", "height")]
        self.actor_rect = {"left": round(x+.13*w), "top": round(y+1.55*h),
                           "width": round(.56*w), "height": round(2.65*h)}
        self.watch_rect = {"left": max(0, round(x-.2*w)), "top": max(0, round(y-h)),
                           "width": round(1.5*w), "height": round(5.6*h)}
        self.watch_rect["width"] = min(self.watch_rect["width"], self.world.shape[1]-self.watch_rect["left"])
        self.watch_rect["height"] = min(self.watch_rect["height"], self.world.shape[0]-self.watch_rect["top"])
        self.idle = bounds(self.world, self.actor_rect)
        if self.idle is None:
            self.fail("玩家靠近画面边缘，无法建立姿势参考；请调整视野后按 F1")
            return
        self.idle = self.idle.copy()
        self.idle_watch = bounds(self.world, self.watch_rect).copy()
        self.wait_pose = self.pose_mask = self.wait_watch = None

    def observe_pose(self, image, now):
        relative = {**self.actor_rect, "left": self.actor_rect["left"]-self.watch_rect["left"],
                    "top": self.actor_rect["top"]-self.watch_rect["top"]}
        pose = bounds(image, relative)
        if pose is None:
            return False, False
        self.pose, self.last_pose_time = pose.copy(), now
        selected = self.pose_mask if self.pose_mask is not None else np.ones(pose.shape[:2], bool)
        idle_difference = float(np.abs(pose.astype(np.float32)-self.idle).mean(2)[selected].mean())
        wait_difference = (float(np.abs(pose.astype(np.float32)-self.wait_pose).mean(2)[selected].mean())
                           if self.wait_pose is not None else 100.)
        # An item held above the player's head is not the idle pose, even when
        # most of the body looks identical. Ignore small animated background changes.
        overhead_height = max(1, relative["top"])
        overhead_change = np.abs(image[:overhead_height].astype(np.float32)
                                 - self.idle_watch[:overhead_height]).mean(2) > 48
        overhead_changed = float(overhead_change.mean()) >= .022
        idle = (idle_difference < 24 and idle_difference < wait_difference * .65
                and not overhead_changed)
        body_changed = wait_difference > 30 if self.wait_pose is not None else idle_difference > 30
        # A bite marker changes pixels above the head. While waiting it must not
        # be interpreted as an item being held up / a completed harvest.
        changed = body_changed or (overhead_changed and self.phase not in ("wait", "settle"))
        self.idle_since = (self.idle_since if self.idle_since is not None else now) if idle else None
        self.changed_since = (self.changed_since if self.changed_since is not None else now) if changed else None
        return (self.idle_since is not None and now-self.idle_since >= .30,
                self.changed_since is not None and now-self.changed_since >= .70)

    def learn_wait_pose(self, watch):
        relative = {**self.actor_rect, "left": self.actor_rect["left"]-self.watch_rect["left"],
                    "top": self.actor_rect["top"]-self.watch_rect["top"]}
        pose = bounds(watch, relative)
        if pose is None:
            return False
        mask = np.abs(pose.astype(np.float32)-self.idle).mean(2) > 16
        if int(mask.sum()) < max(20, pose.shape[0]*pose.shape[1]*.01):
            return False
        self.wait_pose, self.pose_mask = pose.copy(), mask
        if self.wait_watch is None:
            self.wait_watch = watch.copy()
        self.idle_since = self.changed_since = None
        return True

    @staticmethod
    def yellow(image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # Daylight washes out saturation; night tint lowers value. The marker
        # must also be new relative to the same scene, so this can stay broad.
        return cv2.inRange(hsv, (10, 55, 55), (53, 255, 255))

    def bite_evidence(self, watch):
        baseline = self.wait_watch if self.wait_watch is not None else self.idle_watch
        difference = np.abs(watch.astype(np.int16) - baseline.astype(np.int16))
        novel = difference.max(2) >= 18
        novel_since_cast = np.abs(watch.astype(np.int16) - self.idle_watch.astype(np.int16)).max(2) >= 18
        mask = self.yellow(watch)
        # Only this actor's head area is searched, never a global ! template.
        # Quest/UI icons elsewhere on the screen cannot trigger this detector.
        rect = self.power_rect
        cx = rect["left"] + .42*rect["width"] - self.watch_rect["left"]
        ytop = rect["top"] - .65*rect["height"] - self.watch_rect["top"]
        ybottom = rect["top"] + 1.5*rect["height"] - self.watch_rect["top"]
        ys, xs = np.indices(mask.shape)
        mask[(np.abs(xs-cx) > .21*rect["width"]) | (ys < ytop) | (ys > ybottom)] = 0
        # Remove stationary yellow scenery before connected components. On a
        # bright day it otherwise joins the exclamation stem into a wide blob.
        mask[~novel] = 0
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8))
        _, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
        components = stats[1:]
        evidence = {"visible": False, "region": [max(0, round(cx-.21*rect["width"])),
                    max(0, round(ytop)), round(.42*rect["width"]), round(ybottom-ytop)],
                    "stem_candidates": 0, "novel_fraction": 0.,
                    "baseline": "wait" if self.wait_watch is not None else "pre_cast"}
        for i, (x, y, w, h, area) in enumerate(components, 1):
            if not (2.0 <= h/max(w, 1) <= 8 and .22*rect["height"] <= h <= 1.4*rect["height"] and area >= w*h*.45):
                continue
            evidence["stem_candidates"] += 1
            for j, (xx, yy, ww, hh, aa) in enumerate(components, 1):
                if (y+h <= yy <= y+h+max(4, w*2.5) and abs(xx+ww/2-(x+w/2)) <= max(3, w)
                        and .45*w <= ww <= 2*w and .45*w <= hh <= 1.8*w and aa >= 3):
                    novel_fraction = min(float(novel_since_cast[labels == i].mean()),
                                         float(novel_since_cast[labels == j].mean()))
                    if novel_fraction < .60:
                        continue
                    evidence.update(visible=True, stem=list(map(int, (x,y,w,h))),
                                    dot=list(map(int, (xx,yy,ww,hh))), novel_fraction=novel_fraction)
                    return evidence
        return evidence

    def poll_bite(self, watch, hwnd):
        now = time.perf_counter()
        evidence = self.bite_evidence(watch)
        visible = evidence["visible"]
        if visible:
            stem = evidence["stem"]
            center = stem[0] + stem[2]/2
            continuous = (now-self.last_visual <= .10 and self.last_visual_x is not None
                          and abs(center-self.last_visual_x) <= self.power_rect["width"]*.08)
            self.visual_count = self.visual_count+1 if continuous else 1
            self.last_visual, self.last_visual_x = now, center
        else:
            self.visual_count = 0
        event = self.audio.consume_event() if self.audio else None
        if event:
            self.pending_audio = event
        heard = bool(self.pending_audio and 0 <= now-self.pending_audio["time"] <= .30)
        audio_state = self.audio.snapshot() if self.audio else None
        confirmed = (visible and (self.visual_count >= 2 or heard)
                     and self.config.get("auto_bite") != "enchanted")
        self.bite_times.append(self.last_capture)
        hz = ((len(self.bite_times)-1) / max(self.bite_times[-1]-self.bite_times[0], 1e-6)
              if len(self.bite_times) > 1 else 0.)
        record = {"type": "bite_frame", "captured": self.last_capture, "time": now,
                  "cast": self.casts, "phase": self.phase, "watch_rect": dict(self.watch_rect),
                  "evidence": evidence, "visual_frames": self.visual_count,
                  "audio": audio_state, "audio_event": event, "audio_recent": heard,
                  "audio_without_visual": bool(event and not visible),
                  "confirmed": confirmed, "actual_hz": hz}
        with self.session.ring_lock:
            self.session.bite_ring.append((watch, record))
        if visible or event or now-self.last_bite_log >= .25:
            self.session._record(record)
            self.last_bite_log = now
        if now-self.last_bite_preview >= .10:
            with self.session.state_lock:
                self.session.preview = (watch, record)
            self.last_bite_preview = now
        self.session.publish(bite_hz=hz, bite_visible=visible, bite_visual_frames=self.visual_count,
                             bite_audio=audio_state, bite_audio_recent=heard)
        if confirmed and self.generation == self.session.generation and self.session.active.is_set():
            if self.session.mouse.set(True, hwnd):
                self.transition("hook", "头顶新出现 ! + 声音辅助" if heard else "头顶新出现 ! 连续两帧")
                return True
        return False

    def scene_ready(self, frame):
        # Large newly drawn menus must not be accepted as an idle world.
        size = (320, 180)
        a = cv2.GaussianBlur(cv2.resize(frame, size), (7, 7), 0).astype(np.float32)
        b = cv2.GaussianBlur(cv2.resize(self.world, size), (7, 7), 0).astype(np.float32)
        changed = np.abs(a-b).mean(2) > 40
        return float(changed[20:153, 12:285].mean()) < .035

    def begin_cast(self, hwnd, client):
        origin = self.config["auto_origin"]
        self.session.mouse.move(client["left"]+origin["x"], client["top"]+origin["y"], hwnd)
        self.collect_count = 0
        self.casts += 1
        self.cast_started = self.session.mouse.hold_for(CAST_HOLD_SECONDS, hwnd)
        if self.cast_started is None:
            self.fail("未能按下抛竿，请确认游戏仍在前台")
            return
        self.session.publish(auto_hold_elapsed=0., auto_hold_seconds=CAST_HOLD_SECONDS)
        self.transition("charge", f"按用户标定按住 {CAST_HOLD_SECONDS:.2f} 秒后松开")

    def finish_cast_if_released(self):
        if not self.session.active.is_set() or self.generation != self.session.generation:
            return True
        with self.session.mouse.lock:
            if self.session.mouse.holding:
                return False
            released = dict(self.session.mouse.last_release or {})
        if released.get("reason") != "cast_duration_complete":
            self.fail("蓄力输入已被中断")
        elif self.actor_rect is None:
            self.fail("已按 1.05 秒松开，但未定位玩家；请检查界面缩放")
        else:
            self.transition("settle", f"计划 {CAST_HOLD_SECONDS:.2f} 秒，实际按住 {released['held_seconds']:.3f} 秒")
        return True

    def tick(self, sct, client, hwnd):
        if self.generation != self.session.generation or not self.session.active.is_set():
            return
        now = time.perf_counter()
        if self.config["auto_origin"]["client_size"] != [client["width"], client["height"]]:
            self.fail("游戏窗口尺寸改变，请在新位置重新按 F1")
            return
        elapsed = now-self.entered
        if self.phase == "arming":
            if self.world is None:
                self.world = self.capture(sct, client)
                self.energy = self.reader.energy_fraction(self.world)
                proposal = self.session.matcher.locate(self.world)
                check = {"type": "auto_start_check", "time": now, "energy": self.energy,
                         "proposal": proposal, "active_minigame": False}
                if proposal is not None:
                    crop = bounds(self.world, proposal)
                    if crop is not None:
                        from vision_live import decode_geometry
                        structure = self.session.matcher.score(crop)
                        prediction = self.session.vision.predict([self.session.matcher.inference_crop(crop)])[0]
                        # Starting a cast must check the complete bar/progress
                        # layout. Fish-like scenery alone is not a minigame;
                        # a real panel still counts when its fish is occluded.
                        _, _, evidence, reason = decode_geometry(prediction, crop, structure,
                            float(self.config.get("fish_offset_native", 1.)), (), require_fish=False)
                        check.update(active_minigame=not bool(reason), rejection_reason=reason,
                                     presence=prediction["presence_scores"], evidence=evidence)
                        if not reason:
                            self.session._record(check)
                            self.fail("请在当前小游戏结束、玩家空闲时按 F1")
                            return
                self.session._record(check)
            if elapsed >= .4:
                if self.low_energy_confirmed(sct, client):
                    self.fail("精力低于 12%，请先恢复精力")
                else:
                    self.begin_cast(hwnd, client)
        elif self.phase == "charge":
            self.session.publish(auto_hold_elapsed=min(CAST_HOLD_SECONDS, now-self.cast_started))
            if self.finish_cast_if_released():
                return
            # The meter remains a player-position cue only. It never controls
            # the release time; the independent input timer does that.
            if self.power_rect is None:
                frame = self.capture(sct, client)
                located = self.reader.locate_power(frame, self.world)
                if located:
                    self.power_rect, _ = located
                    self.calibrate_actor(self.power_rect)
            if not self.session.active.is_set() or self.generation != self.session.generation:
                return
            self.finish_cast_if_released()
        elif self.phase == "settle":
            if elapsed >= .8:
                watch = self.capture(sct, client, self.watch_rect)
                # Detect an early bite while the settling animation is ending;
                # do not learn an already-visible ! as stationary background.
                if self.poll_bite(watch, hwnd):
                    return
                if self.wait_pose is None and not self.visual_count:
                    self.learn_wait_pose(watch)
                if elapsed >= 1.8:
                    learned = self.wait_pose is not None
                    if not self.visual_count:
                        learned = self.learn_wait_pose(watch) or learned
                    if learned:
                        self.transition("wait", "已建立等待姿势；持续检测玩家头顶 !")
                    elif elapsed > 3.5:
                        self.fail("抛竿后没有确认等待姿势，可能未落水")
        elif self.phase == "wait":
            watch = self.capture(sct, client, self.watch_rect)
            idle, changed = self.observe_pose(watch, now)
            if self.poll_bite(watch, hwnd):
                return
            recent_marker = now-self.last_visual < .60
            if idle and elapsed > 1 and not recent_marker:
                self.transition("cooldown", "玩家已回到空闲姿势")
            elif changed and elapsed > 1 and not recent_marker:
                self.transition("resolve", "等待姿势改变，检查直接收获或收竿")
            elif elapsed > 90:
                self.fail("90 秒内未确认咬钩；请按 F9 保存头顶检测区域，并检查上钩方式")
        elif self.phase in ("hook", "collect"):
            if elapsed < .065:
                self.session.mouse.set(True, hwnd)
            else:
                self.session.mouse.release("auto_click_complete")
                next_phase = "hook_wait" if self.phase == "hook" else "resolve"
                self.transition(next_phase, "单次点击已完成")
        elif self.phase == "hook_wait":
            if elapsed >= 3.5:
                self.transition("resolve", "未出现小游戏，检查杂物/直接收获")
        elif self.phase == "playing":
            if now-self.last_panel > .90:
                self.session.mouse.release("minigame_disappeared")
                self.idle_since = self.changed_since = None
                self.transition("resolve", "面板已持续消失，等待收竿动画")
            elif now-self.last_valid > 3.0:
                self.fail("小游戏仍在但观测持续异常；请按 F9 保存片段")
        elif self.phase == "resolve":
            if now-self.last_panel < .8:
                return
            watch = self.capture(sct, client, self.watch_rect)
            idle, changed = self.observe_pose(watch, now)
            if idle and elapsed > .8:
                self.transition("cooldown", "已确认空闲姿势")
            elif elapsed > 1.6 and changed and self.collect_count < 2:
                self.collect_count += 1
                self.session.mouse.set(True, hwnd)
                self.transition("collect", f"确认收获，第 {self.collect_count} 次")
            elif elapsed > 6:
                self.fail("收竿后状态不明；请处理背包/宝箱界面后重新按 F1")
        elif self.phase == "cooldown" and elapsed >= .65:
            frame = self.capture(sct, client)
            if not self.scene_ready(frame):
                self.fail("出现较大界面变化，请处理菜单/背包后重新按 F1")
                return
            self.energy = self.reader.energy_fraction(frame)
            if self.low_energy_confirmed(sct, client):
                self.fail("精力低于 12%，本轮结束")
                return
            # Refresh the confirmed idle pose for slow lighting changes.
            pose = bounds(frame, self.actor_rect)
            if pose is not None:
                self.idle = pose.copy()
                self.idle_watch = bounds(frame, self.watch_rect).copy()
            self.world = frame
            self.begin_cast(hwnd, client)
        audio_state = "声音辅助正常" if self.audio and self.audio.ok else "声音不可用，视觉检测中" if self.audio else "纯视觉/附魔"
        if self.session.active.is_set():
            self.session.publish(auto_phase=self.phase, auto_cast=self.casts, auto_energy=self.energy,
                status=f"全自动 · 第 {self.casts} 杆 · {self.LABELS[self.phase]} · {audio_state}",
                holding=self.session.mouse.holding, focused=True)
