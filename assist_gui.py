"""Stardew 자동낚시 — 보조모드 GUI (시작/종료 토글 버튼).

사용자가 직접 캐스팅+후킹, 봇은 미니게임 바만 제어. 완전자동 아님.
- [시작] 버튼을 누르면 감시 시작 → 버튼이 [종료]로 바뀜.
- [종료]를 누르면 정지 → 다시 [시작]으로.
- 전역 키후킹 없음 → ESC 등 다른 키/작업으로 절대 자동 종료되지 않음. 버튼으로만 제어.

실행:  pythonw assist_gui.py   (콘솔 없이)  또는  python assist_gui.py
"""
import os
import threading
import time
import ctypes
from ctypes import wintypes
import traceback
from datetime import datetime

import tkinter as tk

import numpy as np
import mss
import pydirectinput
from pynput import mouse as pynput_mouse

from audio_bite import AudioBiteDetector
from main import load_config, grab, detect, BarController, Mouse, _runtime_dir, find_game_hwnd

pydirectinput.PAUSE = 0
pydirectinput.FAILSAFE = False
try:
    ctypes.windll.user32.SetProcessDPIAware()
except Exception:
    pass


class AssistGUI:
    def __init__(self):
        self.cfg = load_config()
        self.running = False
        self.thread = None
        self.status = "정지"          # 워커가 갱신, GUI가 폴링해서 표시
        self.worker_error = None      # 워커가 예외로 죽으면 여기에 사유 기록(GUI가 감지)
        self._build_ui()

    # ---------------------------------------------------------------- UI
    def _build_ui(self):
        self.root = tk.Tk()
        self.root.title("Stardew 자동낚시")
        self.root.resizable(False, False)
        self.root.geometry("250x130+30+30")         # 좌상단. 게임 뒤로 가도 됨(alt-tab으로 종료)

        self.btn = tk.Button(self.root, text="시작", command=self.toggle,
                             font=("맑은 고딕", 18, "bold"), bg="#2ecc71", fg="white",
                             activebackground="#27ae60", relief="flat", height=2)
        self.btn.pack(fill="x", padx=12, pady=(14, 8))

        self.lbl = tk.Label(self.root, text="정지 — [시작]을 누르세요",
                            font=("맑은 고딕", 9), fg="#555")
        self.lbl.pack()

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self._poll()

    def toggle(self):
        if self.running:
            self.stop()
        else:
            self.start()

    def start(self):
        if self.running:
            return
        self.running = True
        self.worker_error = None
        self.status = "대기중"
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()
        self.btn.config(text="종료", bg="#e74c3c", activebackground="#c0392b")

    def stop(self):
        self.running = False                       # 워커가 다음 루프에서 빠져나가며 마우스 놓음
        self.btn.config(text="시작", bg="#2ecc71", activebackground="#27ae60")

    def on_close(self):
        self.running = False
        self.root.after(150, self.root.destroy)

    # ---------------------------------------------------------------- 제어 루프(별도 스레드)
    def _worker(self):
        cfg = self.cfg
        mouse = Mouse()
        use_ppo = cfg.get("controller", "rule") == "ppo"
        ctrl = None
        interval = 1.0 / (30 if use_ppo else max(int(cfg["fps"]), 5))
        bar_grace = float(cfg.get("bar_grace", 0.35))
        min_pulse = 0.0 if use_ppo else float(cfg.get("min_pulse", 0.0))
        confirm_frames = int(cfg.get("confirm_frames", 3))   # 진입 확정에 필요한 연속 프레임
        in_prev = False
        confirm = 0                 # 초록바+물고기 동시 검출 연속 프레임 카운터(진입 게이트)
        last_press = False
        bar_seen_t = 0.0
        last_log = 0.0
        # 미니게임별 통계(문제 진단용): 프레임/물고기검출/추적오차/inside(바 안에 있던 비율)
        mg_frames = 0
        mg_fish = 0
        mg_inside = 0
        mg_errs = []
        mg_start = 0.0
        mg_no = 0
        worker_start_t = time.perf_counter()
        auto_hook = bool(cfg.get("auto_hook", False))
        game_hwnd = find_game_hwnd(cfg.get("game_title", "Stardew"))
        last_window_search = 0.0
        last_wait_log = 0.0
        cast_down_t = None
        cast_ready_t = 0.0
        hook_pending_until = 0.0
        tracking_cast = False
        cast_lock = threading.Lock()
        listener = None
        audio = None

        # ---- 로그 파일 열기(항상 기록). logs/assist_YYYYMMDD_HHMMSS.log ----
        log = self._open_log(cfg)

        def drive(now, bar, fish, frame, progress_frame, progress, vision_ms):
            press = ctrl.update(now, bar, fish, progress, vision_ms) if use_ppo else ctrl.update(now, bar, fish)
            # Check again immediately before input, after capture and inference.
            if not self.running or ctypes.windll.user32.GetForegroundWindow() != game_hwnd:
                mouse.up()
            else:
                mouse.set(press, now, min_pulse)
            if use_ppo:
                ctrl.input_applied(mouse.holding, time.perf_counter())
                ctrl.snapshot(frame, progress_frame, now)
            return mouse.holding

        def tracking_status(bar, fish):
            if use_ppo:
                return ctrl.status
            return "미니게임 제어중  bar=%d fish=%s" % (int(bar["center"]), int(fish["center"]) if fish else None)

        def on_click(_x, _y, button, pressed):
            nonlocal cast_down_t, cast_ready_t, hook_pending_until
            if button != pynput_mouse.Button.left or not tracking_cast or not game_hwnd:
                return
            now = time.perf_counter()
            with cast_lock:
                if pressed:
                    rect = wintypes.RECT()
                    if ctypes.windll.user32.GetWindowRect(game_hwnd, ctypes.byref(rect)) and (
                        rect.left <= _x < rect.right and rect.top <= _y < rect.bottom
                    ):
                        cast_down_t = now
                elif cast_down_t is not None:
                    held = now - cast_down_t
                    cast_down_t = None
                    if held >= 0.12 and ctypes.windll.user32.GetForegroundWindow() == game_hwnd:
                        hook_pending_until = 0.0
                        cast_ready_t = now + float(cfg.get("cast_settle", 1.7))
                        log("   [手动抛竿] 按住 %.2f 秒，%.1f 秒后开始监听咬钩" %
                            (held, float(cfg.get("cast_settle", 1.7))))

        def summarize():
            if mg_frames <= 0:
                return
            dur = time.perf_counter() - mg_start
            if mg_errs:
                arr = sorted(mg_errs)
                med = arr[len(arr) // 2]
                mx = arr[-1]
                det = 100.0 * mg_fish / mg_frames
                inside = 100.0 * mg_inside / max(mg_fish, 1)   # 물고기가 바 안에 있던 비율
                # inside 로 잡음/놓침 추정(확정 아님 — 추후 진행바 판독으로 대체 예정)
                hint = "잡음추정" if inside >= 60 else ("애매" if inside >= 40 else "놓침추정")
                log("  << 미니게임#%d 종료 — %.1f초 %d프레임, 물고기검출 %.0f%%, inside %.0f%% [%s], 추적오차 median=%.0f max=%.0f"
                    % (mg_no, dur, mg_frames, det, inside, hint, med, mx))
            else:
                log("  << 미니게임#%d 종료 — %.1f초 %d프레임, 물고기 검출 0%% (색/위치 확인 필요)"
                    % (mg_no, dur, mg_frames))

        err_streak = 0                              # 연속 프레임 오류 카운터
        try:
            if use_ppo:
                self.status = "正在加载 1000 万步 PPO 模型…"
                from ppo_live import PPOController, detect_progress
                ctrl = PPOController(cfg, log=log, log_path=self.logpath)
            else:
                ctrl = BarController(cfg)
            if auto_hook:
                audio = AudioBiteDetector(cfg, log=log)
                listener = pynput_mouse.Listener(on_click=on_click)
                listener.start()
            with mss.mss() as sct:
                while self.running:
                    t0 = time.perf_counter()
                    try:
                        if game_hwnd and not ctypes.windll.user32.IsWindow(game_hwnd):
                            game_hwnd = None
                        if game_hwnd is None and t0 - last_window_search >= 1.0:
                            game_hwnd = find_game_hwnd(cfg.get("game_title", "Stardew"))
                            last_window_search = t0
                        focused = bool(game_hwnd and ctypes.windll.user32.GetForegroundWindow() == game_hwnd)
                        if not focused:
                            tracking_cast = auto_hook
                            if audio is not None:
                                audio.arm(False)
                            if in_prev:
                                summarize()
                                in_prev = False
                            confirm = 0
                            ctrl.reset()
                            mouse.up()
                            self.status = ("未找到星露谷窗口" if game_hwnd is None else "请切回星露谷游戏窗口")
                            time.sleep(interval)
                            continue

                        vision_started = time.perf_counter()
                        frame = grab(sct, cfg["roi"])
                        bar, fish, _, _ = detect(frame, cfg)
                        progress_frame = grab(sct, cfg["progress_roi"]) if use_ppo else None
                        progress = detect_progress(progress_frame) if use_ppo else (None, 0.0)
                        vision_ms = (time.perf_counter() - vision_started) * 1000

                        if audio is not None:
                            with cast_lock:
                                ready = cast_ready_t > 0 and t0 >= cast_ready_t
                            audio.arm(ready and not in_prev)

                        if not in_prev:
                            tracking_cast = auto_hook and t0 >= hook_pending_until
                            # ---- 아직 미니게임 아님: 진입 게이트만 평가(마우스 절대 안 누름) ----
                            # 진짜 미니게임 = 초록 바 + 청록 물고기가 '함께' 있음. 메뉴/배경 초록이나
                            # 물고기 없는 정적 초록(phantom)에 낚이지 않도록, 둘 다 연속 검출돼야 확정.
                            if bar is not None and fish is not None and (not use_ppo or progress[0] is not None):
                                confirm += 1
                            else:
                                confirm = 0
                            if confirm >= confirm_frames:
                                in_prev = True
                                tracking_cast = False
                                hook_pending_until = 0.0
                                if audio is not None:
                                    audio.arm(False)
                                with cast_lock:
                                    cast_ready_t = 0.0
                                confirm = 0
                                mg_no += 1
                                mg_frames = 0
                                mg_fish = 0
                                mg_inside = 0
                                mg_errs = []
                                mg_start = t0
                                bar_seen_t = t0
                                log(">> 미니게임#%d 시작" % mg_no)
                                # 확정 프레임부터 바로 제어
                                press = drive(t0, bar, fish, frame, progress_frame, progress, vision_ms)
                                last_press = press
                                mg_frames += 1
                                bc = int(bar["center"])
                                fy = int(fish["center"])
                                mg_fish += 1
                                mg_errs.append(abs(bc - fy))
                                if bar["top"] <= fish["center"] <= bar["bottom"]:
                                    mg_inside += 1
                                self.status = tracking_status(bar, fish)
                            else:
                                # 미확정 — phantom일 수 있으니 마우스 놓고 대기
                                ctrl.reset()
                                mouse.up()
                                if use_ppo and bar is not None and fish is not None and progress[0] is None:
                                    self.status = "已识别鱼和绿条，但未识别右侧进度；请检查进度槽选区"
                                elif bar is not None and fish is None:
                                    self.status = "已识别绿色条，但未识别鱼标记；请检查选区"
                                elif bar is None and fish is not None:
                                    self.status = "已识别鱼标记，但未识别绿色条；请检查选区"
                                elif confirm:
                                    self.status = "正在确认钓鱼小游戏…"
                                elif t0 < hook_pending_until:
                                    self.status = "已自动上钩，等待钓鱼小游戏…"
                                elif hook_pending_until > 0:
                                    self.status = "上钩后未识别小游戏；请检查选区或鱼是否逃脱"
                                elif auto_hook and audio is not None:
                                    with cast_lock:
                                        ready_at = cast_ready_t
                                    if not audio.ok and t0 - worker_start_t > 3.0:
                                        self.status = "音效监听不可用；请手动上钩"
                                    elif ready_at <= 0:
                                        self.status = "待抛竿：请在游戏中按住左键抛竿"
                                    elif t0 < ready_at:
                                        self.status = "已抛竿，等待鱼漂落水…"
                                    elif audio.consume_spike():
                                        tracking_cast = False
                                        audio.arm(False)
                                        with cast_lock:
                                            cast_ready_t = 0.0
                                        if self.running and ctypes.windll.user32.GetForegroundWindow() == game_hwnd:
                                            mouse.click(hold=0.05)
                                        hook_pending_until = time.perf_counter() + float(cfg.get("minigame_grace", 1.6))
                                        log("   [自动上钩] 检测到咬钩音效，已点击左键")
                                        self.status = "已自动上钩，等待钓鱼小游戏…"
                                    else:
                                        threshold = max(
                                            float(cfg.get("bite_sound_floor", 0.03)),
                                            audio.baseline * float(cfg.get("bite_sound_ratio", 3.5)),
                                        )
                                        self.status = "等待咬钩音效（音量 %.3f / 阈值 %.3f）" % (
                                            audio.level, threshold)
                                else:
                                    self.status = "待命中：请手动抛竿，咬钩时点击左键"
                                if t0 - last_wait_log >= 2.0 and (bar is not None or fish is not None or auto_hook):
                                    log("   [待命识别] bar=%s fish=%s progress=%s confidence=%.3f audio=%s level=%.4f baseline=%.4f" %
                                        (int(bar["center"]) if bar else None,
                                         int(fish["center"]) if fish else None,
                                         progress[0] if use_ppo else None, progress[1] if use_ppo else 0.0,
                                         audio.ok if audio is not None else None,
                                         audio.level if audio is not None else 0.0,
                                         audio.baseline if audio is not None else 0.0))
                                    last_wait_log = t0
                        elif bar is not None:
                            # ---- 확정된 미니게임 진행 중 ----
                            # focus_window 안 함: 보조모드는 게임이 이미 포그라운드. 재포커스하면 freeze.
                            bar_seen_t = t0
                            press = drive(t0, bar, fish, frame, progress_frame, progress, vision_ms)
                            last_press = press
                            mg_frames += 1
                            bc = int(bar["center"])
                            fy = int(fish["center"]) if fish else None
                            if fy is not None:                    # 진행 중 물고기 순간 미검출은 허용(빠른물고기)
                                mg_fish += 1
                                mg_errs.append(abs(bc - fy))
                                if bar["top"] <= fish["center"] <= bar["bottom"]:
                                    mg_inside += 1                # 물고기가 초록 바 안에 있음
                            self.status = tracking_status(bar, fish)
                            if t0 - last_log >= 0.1:              # 0.1초 간격 상세 로그
                                log("   bar=%d fish=%s vel=%.0f pred=%d %s"
                                    % (bc, fy, ctrl.last_vel, int(ctrl.last_pred),
                                       "UP" if press else "DOWN"))
                                last_log = t0
                        elif (t0 - bar_seen_t) < bar_grace:
                            # 순간 끊김(초록 flicker) — 리셋하지 말고 직전 동작 유지(연속성 보존)
                            if use_ppo:
                                last_press = drive(t0, bar, fish, frame, progress_frame, progress, vision_ms)
                                self.status = ctrl.status
                            else:
                                mouse.set(last_press, t0, min_pulse)
                        else:
                            in_prev = False
                            confirm = 0
                            summarize()                           # 미니게임 종료 요약
                            ctrl.reset()
                            mouse.up()
                            tracking_cast = auto_hook
                            self.status = ("待抛竿：请在游戏中按住左键抛竿" if auto_hook else
                                           "待命中：请手动抛竿，咬钩时点击左键")
                        err_streak = 0
                    except Exception as e:
                        # 프레임 단위 오류는 세션을 죽이지 않고 로그만 남기고 계속(일시적 글리치 방어).
                        err_streak += 1
                        log("   [프레임 오류 %d] %s: %s" % (err_streak, type(e).__name__, e))
                        try:
                            mouse.up()                            # 오류 시 마우스는 반드시 놓아 안전 확보
                            if ctrl is not None:
                                ctrl.reset()
                        except Exception:
                            pass
                        if err_streak >= 30:                      # ~0.5초 연속 오류면 세션 중단
                            self.worker_error = "연속 프레임 오류 %d회 (%s)" % (err_streak, type(e).__name__)
                            log("   [치명] " + self.worker_error + " — 세션 중단\n" + traceback.format_exc())
                            break
                    el = time.perf_counter() - t0
                    if el < interval:
                        time.sleep(interval - el)
        except Exception as e:
            # while 루프 밖(mss 초기화 등) 예외 — 이전엔 여기서 조용히 죽어 버튼만 [종료]로 남았음.
            self.worker_error = "%s: %s" % (type(e).__name__, e)
            log("[치명적 오류] 워커 중단\n" + traceback.format_exc())
        finally:
            if audio is not None:
                audio.arm(False)
                audio.stop()
            if listener is not None:
                listener.stop()
            if in_prev:
                summarize()
            try:
                mouse.up()                          # 정지 시 반드시 버튼 떼기
            except Exception:
                pass
            if use_ppo and ctrl is not None:
                try:
                    ctrl.close()
                except OSError as exc:
                    log("[PPO] 关闭诊断文件时出错：%s" % exc)
            tail = "" if not self.worker_error else "  (오류: %s)" % self.worker_error
            log("[%s] 세션 종료%s" % (datetime.now().strftime("%H:%M:%S"), tail))
            self._close_log()
            self.status = "정지" if not self.worker_error else ("오류 중단: %s" % self.worker_error)

    # ---------------------------------------------------------------- 로깅
    def _open_log(self, cfg):
        """세션 로그 파일을 열고 로그 함수를 반환. 항상 기록(문제 진단용)."""
        try:
            logdir = _runtime_dir() / "logs"
            os.makedirs(logdir, exist_ok=True)
            path = logdir / ("assist_%s.log" % datetime.now().strftime("%Y%m%d_%H%M%S"))
            self._logf = open(path, "w", encoding="utf-8")
            self.logpath = str(path)
        except Exception:
            self._logf = None
            self.logpath = None

        def log(msg):
            if self._logf is None:
                return
            try:
                self._logf.write(msg + "\n")
                self._logf.flush()
            except Exception:
                pass

        params = {k: cfg.get(k) for k in ("controller", "ppo_model", "ppo_level", "ppo_tackle", "progress_roi", "lookahead", "vel_window", "min_pulse", "bar_grace", "aim_offset", "fps", "roi", "fish_margin_bottom", "auto_hook")}
        log("[%s] 세션 시작  params=%s" % (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), params))
        return log

    def _close_log(self):
        try:
            if getattr(self, "_logf", None):
                self._logf.close()
        except Exception:
            pass
        self._logf = None

    # ---------------------------------------------------------------- 상태 표시 폴링
    def _poll(self):
        if self.running:
            # 워커 스레드가 예외로 조용히 죽었는지 감지 → 버튼/상태 복구(안 그러면 [종료]로 멈춰 보임).
            if self.thread is not None and not self.thread.is_alive():
                self.running = False
                self.thread = None
                self.btn.config(text="시작", bg="#2ecc71", activebackground="#27ae60")
                msg = self.worker_error or "알 수 없는 오류"
                self.lbl.config(text="오류로 멈춤: %s — 로그 확인 후 다시 [시작]" % msg, fg="#c0392b")
            else:
                self.lbl.config(text=self.status, fg="#111")
        else:
            self.lbl.config(text="정지 — [시작]을 누르세요", fg="#555")
        self.root.after(120, self._poll)

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    AssistGUI().run()
