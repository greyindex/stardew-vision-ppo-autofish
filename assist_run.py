"""미니게임 보조 모드 (헤드리스). 봇은 캐스팅/후킹 안 함 — 사용자가 직접 던지고 후킹.
미니게임 바가 뜨면 봇이 바를 물고기에 맞춰 제어. 물고기 추적 수정 검증용.
매 프레임 로깅 + roi_dbg 스냅샷. ESC 종료."""
import time, ctypes
from pathlib import Path
import numpy as np, mss, cv2
import pydirectinput
from pynput import keyboard
from main import load_config, grab, detect, BarController, Mouse, find_game_hwnd, focus_window

pydirectinput.PAUSE = 0; pydirectinput.FAILSAFE = False
try: ctypes.windll.user32.SetProcessDPIAware()
except Exception: pass

SCR = Path(__file__).resolve().parent / "logs"
SCR.mkdir(exist_ok=True)
cfg = load_config()
mouse = Mouse(); ctrl = BarController(cfg)
hwnd = find_game_hwnd(cfg.get("game_title", "Stardew"))
interval = 1.0 / max(int(cfg["fps"]), 5)
quit = {"q": False}
def on_key(k):
    if k == keyboard.Key.esc: quit["q"] = True; return False
keyboard.Listener(on_press=on_key).start()

bar_grace = float(cfg.get("bar_grace", 0.35))   # bar=None 이 이 시간 안이면 직전 동작 유지(끊김 무시)
print("[assist] 보조모드. 직접 던지고 '!' 때 후킹하세요. 미니게임 바 뜨면 봇이 제어. ESC종료.", flush=True)
snap_n = 0; snap_f = 0; last = 0.0; in_prev = False; last_press = False; bar_seen_t = 0.0
with mss.mss() as sct:
    while not quit["q"]:
        t0 = time.time()
        frame = grab(sct, cfg["roi"])
        bar, fish, _, _ = detect(frame, cfg)
        if bar is not None:
            bar_seen_t = t0
            if hwnd and ctypes.windll.user32.GetForegroundWindow() != hwnd:
                focus_window(hwnd)
            press = ctrl.update(t0, bar, fish)
            last_press = press
            mouse.set(press, t0, float(cfg.get("min_pulse", 0.0)))
            if not in_prev:
                print("[assist] >> 미니게임 시작", flush=True); in_prev = True
            view = np.ascontiguousarray(frame)
            cv2.rectangle(view, (0, int(bar["top"])), (view.shape[1]-1, int(bar["bottom"])), (0,255,0), 2)
            if fish: cv2.circle(view, (view.shape[1]//2, int(fish["center"])), 7, (0,0,255), 2)
            snap_f += 1
            if snap_n < 12 and snap_f % 12 == 0:
                cv2.imwrite(str(SCR / ("assist_%02d.png" % snap_n)), cv2.resize(view, (view.shape[1]*5, view.shape[0]), interpolation=cv2.INTER_NEAREST))
                snap_n += 1
            if t0 - last > 0.2:
                fy = int(fish["center"]) if fish else None
                print("[assist] bar=%d fish=%s vel=%.0f pred=%d %s" % (
                    int(bar["center"]), fy, ctrl.last_vel, int(ctrl.last_pred),
                    "UP누름" if press else "DOWN뗌"), flush=True)
                last = t0
        elif in_prev and (t0 - bar_seen_t) < bar_grace:
            # 순간 끊김(초록 flicker) — 리셋하지 말고 직전 동작 유지. 제어 연속성 보존.
            mouse.set(last_press, t0, float(cfg.get("min_pulse", 0.0)))
        else:
            if in_prev:
                print("[assist] << 미니게임 종료", flush=True); in_prev = False
            ctrl.reset(); mouse.up()
        el = time.time() - t0
        if el < interval: time.sleep(interval - el)
mouse.up()
print("[assist] 종료.", flush=True)
