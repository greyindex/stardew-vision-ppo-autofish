"""
Stardew Valley 자동 낚시 봇 (완전 자동)
=======================================
캐스팅 → 입질 감지 → 후킹 → 미니게임까지 전부 자동으로 처리합니다.

사용법:
  1) 최초 1회 캘리브레이션 (낚시 미니게임이 떠 있는 상태에서 실행)
       python main.py calibrate         # 미니게임 트랙 영역 선택
       python main.py calibrate-bite    # 입질(!) 감지 영역 선택 (선택)

  2) 색상/마스크 확인이 필요하면 (선택)
       python main.py tune

  3) 봇 실행
       python main.py run
       - F8  : 봇 일시정지 / 재개
       - F9  : 완전자동(캐스팅+입질+미니게임) ON/OFF
       - ESC : 종료

기본 좌표는 1920x1080 창모드 기준으로 맞춰져 있습니다. 창 위치/크기가 다르면
calibrate로 다시 잡아주세요.
"""

import sys
import json
import time
import ctypes
from pathlib import Path

import cv2
import numpy as np
import mss
import pydirectinput
from pynput import keyboard


# ---------------------------------------------------------------- 게임 창 포커스 (입력 전달용)

def find_game_hwnd(title_substr="stardew"):
    """제목에 title_substr 가 들어간 보이는 창의 hwnd 반환 (없으면 None)."""
    user32 = ctypes.windll.user32
    found = []

    @ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)
    def cb(hwnd, _lparam):
        if not user32.IsWindowVisible(hwnd):
            return True
        n = user32.GetWindowTextLengthW(hwnd)
        if n <= 0:
            return True
        buf = ctypes.create_unicode_buffer(n + 1)
        user32.GetWindowTextW(hwnd, buf, n + 1)
        if title_substr.lower() in buf.value.lower():
            found.append(hwnd)
            return False
        return True

    user32.EnumWindows(cb, 0)
    return found[0] if found else None


def focus_window(hwnd):
    """창을 최상위(포그라운드)로 올린다. 포그라운드 제약 회피를 위해 ALT 트릭 사용.

    주의: 최대화/전체화면(borderless) 창에 SW_RESTORE 를 걸면 창모드로 풀려
    좌표가 전부 어긋난다. 따라서 '최소화됐을 때(IsIconic)'만 복원한다.
    """
    if not hwnd:
        return False
    user32 = ctypes.windll.user32
    if user32.IsIconic(hwnd):
        user32.ShowWindow(hwnd, 9)  # SW_RESTORE (최소화 상태일 때만)
    user32.keybd_event(0x12, 0, 0, 0)      # ALT down
    ok = bool(user32.SetForegroundWindow(hwnd))
    user32.keybd_event(0x12, 0, 2, 0)      # ALT up
    return ok

def _runtime_dir() -> Path:
    """실행 파일이 있는 폴더. 소스 실행=이 파일 폴더, PyInstaller exe=exe 폴더."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent
    return Path(__file__).parent


CONFIG_PATH = _runtime_dir() / "config.json"        # 읽기/쓰기 대상(exe 옆 또는 소스 폴더)

DEFAULT_CONFIG = {
    # --- 미니게임 트랙(초록 바가 오르내리는 세로 물기둥) 영역: 1920x1080 기준 ---
    "roi": {"left": 802, "top": 206, "width": 31, "height": 667},

    # --- 입질(!) 감지 영역 (찌 위쪽 물 영역): 1920x1080 기준 ---
    "bite_roi": {"left": 895, "top": 770, "width": 160, "height": 120},

    # --- 초록 캐치 바 HSV 범위 (OpenCV H:0~179) ---
    "bar_hsv_lower": [33, 100, 90],
    "bar_hsv_upper": [70, 255, 255],

    # --- 물고기 표식(청록색) HSV 범위. H~96 청록. 바닥의 파란 물(H~120)과 색으로 구분 ---
    "fish_hsv_lower": [86, 110, 100],
    "fish_hsv_upper": [108, 255, 245],

    "bar_min_area": 400,   # 초록 바로 인정할 최소 픽셀 면적 (해초 오검출 방지)
    "fish_min_area": 8,    # 물고기로 인정할 최소 픽셀 면적
    # 트랙 위/아래 가장자리(프레임) 살짝 제외 (색으로 이미 구분되므로 크게 필요 없음)
    "fish_margin_top": 8,
    "fish_margin_bottom": 8,

    # --- 미니게임 제어 (예측/속도 기반) ---
    "lookahead": 0.12,  # 예측 시간(초). 클수록 물고기에 닿기 전에 더 일찍 떼서 덜 튐(=클릭 덜함).
                        #  위로 지나쳐 튀면 ↑, 안 올라가고 못 따라가면 ↓(=더 많이 누름)
    "vel_window": 0.11, # 바 속도를 이 구간(초)으로 추정. 검출 떨림에 강하게(들썩임 방지).
                        #  바가 덜덜 떨며 안 올라가면 ↑, 반응이 느리면 ↓
    "aim_offset": 0.0,  # 조준 보정(px). +면 물고기를 바 중앙보다 살짝 아래로 조준
    "min_pulse": 0.03,  # 최소 클릭/뗌 유지 시간(초). 너무 짧으면 게임이 '누름'을 놓칠 수 있고,
                        #  너무 길면 클릭이 뜸해져 안 올라감. 안 올라가면 ↓, 덜덜 떨면 ↑
    "fps": 60,          # 초당 처리 횟수 (높을수록 속도 추정이 정확)

    # --- 입력 전달: 게임 창 자동 포커스 (클릭이 게임에 안 먹힐 때 필수) ---
    "auto_focus": True,
    "game_title": "Stardew",

    # --- 완전 자동 파라미터 ---
    "full_auto": False,
    "cast_target": {"x": 960, "y": 900},  # 캐스팅 전 커서를 옮길 물 위 지점(1080 기준)
    "cast_power": 0.12,      # 캐스팅 시 마우스 버튼 유지 시간(초). 길수록 멀리 던짐
    "cast_settle": 1.7,      # 캐스팅 후 찌가 안정될 때까지 대기(초) → 이후 입질 감시 시작
    "wait_timeout": 30,      # 입질 없이 이 시간(초) 지나면 다시 캐스팅

    # --- 입질 감지 방식 ---
    "bite_mode": "sound",    # "sound"=입질 소리 감지(권장) / "vision"=화면 '!' 변화 감지
    #  소리 감지: 대기 중 순간 음량이 (배경 x ratio) 와 floor 를 동시에 넘으면 입질로 판정
    "bite_sound_ratio": 3.5, # 배경 음량 대비 몇 배로 튀어야 입질로 볼지
    "bite_sound_floor": 0.03,# 최소 절대 음량(이보다 조용하면 무시)
    "bite_refractory": 1.0,  # 입질 판정 후 이 시간(초) 동안 재판정 금지
    #  화면 감지(vision): 감지영역 픽셀 중 이 비율 이상 급변하면 '!'로 간주
    "bite_change": 0.16,
    "minigame_grace": 1.6,   # 후킹 후 미니게임이 뜰 때까지 기다리는 시간(초)
    "post_catch_delay": 1.3, # 미니게임 종료 후 보상창 닫고 다음 캐스팅까지 대기(초)
}


# ---------------------------------------------------------------- 설정 IO

def load_config() -> dict:
    cfg = DEFAULT_CONFIG.copy()
    # 우선순위: 실행폴더(exe/소스 옆) config.json → (frozen) 번들 내장 config.json → 기본값
    paths = [CONFIG_PATH]
    if getattr(sys, "frozen", False):
        paths.append(Path(getattr(sys, "_MEIPASS", ".")) / "config.json")
    for p in paths:
        try:
            if p.exists():
                cfg.update(json.loads(p.read_text(encoding="utf-8")))
                break
        except Exception as e:
            print(f"[경고] config.json 읽기 실패, 기본값 사용: {e}")
    return cfg


def save_config(cfg: dict):
    CONFIG_PATH.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[저장] {CONFIG_PATH}")


# ---------------------------------------------------------------- 화면 인식

def grab(sct, roi):
    """ROI 영역 스크린샷 -> BGR ndarray"""
    raw = sct.grab({"left": roi["left"], "top": roi["top"],
                    "width": roi["width"], "height": roi["height"]})
    return np.array(raw)[:, :, :3]  # BGRA -> BGR


def find_blob_center_y(mask, min_area=30):
    """마스크에서 가장 큰 덩어리의 세로 중심 y와 (top, bottom)을 반환"""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(c)
    if area < min_area:
        return None
    x, y, w, h = cv2.boundingRect(c)
    return {"center": y + h / 2, "top": y, "bottom": y + h, "area": area}


def find_bar_center_y(mask, min_area=400):
    """초록 캐치 바 중심 y. 물고기가 바를 가로지르면 초록이 위/아래 두 조각으로
    쪼개진다 → 가장 큰 조각만 고르면 중심이 위아래로 튄다. 그래서 충분히 큰 초록
    조각들을 '합쳐' 전체 세로 범위(최상단~최하단)의 중점을 바 중심으로 쓴다."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    tops, bottoms, total = [], [], 0.0
    piece_min = max(80, int(min_area * 0.35))   # 바 조각으로 볼 최소 면적(작은 노이즈 제외)
    for c in contours:
        a = cv2.contourArea(c)
        if a < piece_min:
            continue
        x, y, w, h = cv2.boundingRect(c)
        tops.append(y)
        bottoms.append(y + h)
        total += a
    if not tops or total < min_area:
        return None
    top = min(tops)
    bottom = max(bottoms)
    return {"center": (top + bottom) / 2.0, "top": top, "bottom": bottom, "area": total}


def find_bar_geometry(mask, fish, min_area=400):
    """Find a wide vertical bar without merging the fish or bottom decoration.

    The real bar becomes translucent outside the fish's hitbox. Its expanded
    color range also includes parts of the fish and the seaweed, so color alone
    is insufficient. Join separated wide rows only across the detected fish.
    """
    height, width = mask.shape
    occupied = (np.count_nonzero(mask, axis=1) >= max(3, width * 0.45))
    edges = np.diff(np.r_[0, occupied, 0].astype(np.int16))
    runs = list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))
    merged = []
    for top, bottom in runs:
        if merged:
            previous_bottom = merged[-1][1]
            small_gap = top - previous_bottom <= max(2, round(height * 0.004))
            fish_gap = (fish is not None and
                        previous_bottom >= fish["top"] - 2 and
                        top <= fish["bottom"] + 2 and
                        top - previous_bottom <= height * 0.08)
            if small_gap or fish_gap:
                merged[-1][1] = int(bottom)
                continue
        merged.append([int(top), int(bottom)])
    candidates = []
    for top, bottom in merged:
        area = int(np.count_nonzero(mask[top:bottom]))
        if 0.10 * height <= bottom - top <= 0.55 * height and area >= min_area:
            candidates.append((area, top, bottom))
    if not candidates:
        return None
    area, top, bottom = max(candidates)
    return {"center": (top + bottom) / 2.0, "top": top, "bottom": bottom, "area": area}


def detect(frame, cfg):
    """트랙 프레임에서 (bar, fish, bar_mask, fish_mask)를 반환.

    bar : 초록 캐치 바 (초록 HSV 범위, H~43)
    fish: 물고기 표식 (청록 HSV 범위, H~96). 바닥의 어두운 파란 물(H~120)과는
          색상(Hue)으로 구분된다.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    bar_mask = cv2.inRange(hsv, np.array(cfg["bar_hsv_lower"]), np.array(cfg["bar_hsv_upper"]))

    # 물고기: 청록색 표식만 (파란 배경/바닥물은 H가 달라 제외됨)
    fish_mask = cv2.inRange(hsv, np.array(cfg["fish_hsv_lower"]), np.array(cfg["fish_hsv_upper"]))
    fish_mask = cv2.morphologyEx(fish_mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    # 위/아래 가장자리(프레임)만 살짝 제외
    mt = int(cfg.get("fish_margin_top", 0))
    mb = int(cfg.get("fish_margin_bottom", 0))
    if mt > 0:
        fish_mask[:mt, :] = 0
    if mb > 0:
        fish_mask[fish_mask.shape[0] - mb:, :] = 0
    fish = find_fish_blob(fish_mask, min_area=int(cfg["fish_min_area"]))
    if cfg.get("controller") == "ppo":
        # Saved real-game misses have H=67..80, S=86..113: the old S>=100,
        # H<=70 mask either lost the bar or cut off its lower half.
        faded_mask = cv2.inRange(hsv, np.array([33, 60, 70]), np.array([85, 255, 255]))
        bar_mask = cv2.bitwise_or(bar_mask, faded_mask)
        bar = find_bar_geometry(bar_mask, fish, min_area=int(cfg["bar_min_area"]))
    else:
        bar = find_bar_center_y(bar_mask, min_area=int(cfg["bar_min_area"]))
    return bar, fish, bar_mask, fish_mask


def find_fish_blob(mask, min_area=8):
    """물고기 청록 덩어리 중심 y. 트랙 바닥의 물 베이스(하단 가장자리에 닿는 큰 청록
    덩어리)는 물고기가 아니므로 제외하고, 남은 것 중 가장 큰 것을 고른다.
    (일부 낚시터는 바닥 물이 물고기와 같은 청록색이라 색만으로는 구분 안 됨)"""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h_roi = mask.shape[0]
    best = None
    best_area = 0.0
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(c)
        if y + h >= h_roi - 4:   # 바닥 가장자리에 닿음 = 물 베이스 → 제외
            continue
        if area > best_area:
            best_area = area
            best = {"center": y + h / 2, "top": y, "bottom": y + h, "area": area}
    return best


def bite_change_fraction(base, cur):
    """두 프레임 간 '크게 변한 픽셀 비율' (0~1). '!' 말풍선이 뜨면 급증한다."""
    if base is None or cur is None or base.shape != cur.shape:
        return 0.0
    d = cv2.absdiff(base, cur)
    return float((d.max(axis=2) > 45).mean())


# ---------------------------------------------------------------- 입력 (마우스)

class Mouse:
    """마우스 버튼 상태 래퍼. 최소 펄스 폭을 두어 너무 빠른 down/up 토글을
    막는다(게임이 '누름 유지'를 확실히 인식하도록)."""

    def __init__(self):
        self.holding = False
        self._last_change = -1.0
        self.downs = 0   # 실제 mouseDown 호출 횟수 (진단용)
        self.ups = 0     # 실제 mouseUp 호출 횟수

    def _apply(self, down, now):
        if down:
            pydirectinput.mouseDown()
            self.downs += 1
        else:
            pydirectinput.mouseUp()
            self.ups += 1
        self.holding = down
        self._last_change = now

    def set(self, want_down, now, min_pulse=0.0):
        """원하는 버튼 상태로. 단, 마지막 변경 후 min_pulse 초가 안 지났으면 유지."""
        if want_down == self.holding:
            return
        if min_pulse > 0 and self._last_change >= 0 and (now - self._last_change) < min_pulse:
            return  # 아직 펄스 유지 시간 → 상태 바꾸지 않음
        self._apply(want_down, now)

    def down(self):
        if not self.holding:
            self._apply(True, self._last_change)

    def up(self):
        if self.holding:
            pydirectinput.mouseUp()
            self.holding = False

    def click(self, hold=0.05):
        pydirectinput.mouseDown()
        time.sleep(hold)
        pydirectinput.mouseUp()
        self.holding = False

    def move(self, x, y):
        pydirectinput.moveTo(int(x), int(y))


# ---------------------------------------------------------------- 미니게임 바 제어 (PWM)

class BarController:
    """물고기를 초록 바에 유지하는 예측(속도 기반) 제어기.

    낚시 바는 관성이 있는 2차계다: 클릭 유지=위로 가속, 떼면 중력으로 하강.
    - 바가 물고기보다 아래면 클릭을 유지해 올린다(연속 누름 → 확실히 상승).
    - 단, 바의 상승 속도를 재서 'lookahead 초 뒤 예상 위치'가 이미 물고기에
      닿거나 지나칠 것 같으면 미리 뗀다 → 관성 오버슈트(위로 튐) 방지.
    이렇게 하면 확실히 올라가면서도 지나치지 않는다. lookahead 가 클수록 더 일찍 뗀다.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.reset()

    def reset(self):
        self.hist = []        # 최근 (시각, 바 중심 y) 기록 → 구간 속도 추정용
        self.last = False     # 직전 클릭 여부
        self.last_vel = 0.0   # 진단용
        self.last_pred = 0.0

    def update(self, now, bar, fish):
        """이번 순간 클릭을 눌러야 하는지(bool)."""
        if bar is None:
            self.reset()
            return False
        bc = bar["center"]
        # 속도는 프레임 단위가 아니라 최근 vel_window(초) 구간으로 추정한다.
        # 검출값이 프레임마다 몇 px씩 떨려도 구간 평균이라 노이즈에 강하다.
        self.hist.append((now, bc))
        win = float(self.cfg.get("vel_window", 0.11))
        while len(self.hist) > 1 and now - self.hist[0][0] > win:
            self.hist.pop(0)

        if fish is None:
            return self.last  # 물고기가 바에 가려 안 보임 → 직전 동작 유지

        t0, b0 = self.hist[0]
        vel = (bc - b0) / max(now - t0, 1e-3)  # px/s, 음수=위로
        target = fish["center"] + float(self.cfg.get("aim_offset", 0.0))
        look = float(self.cfg.get("lookahead", 0.20))
        pred = bc + vel * look               # lookahead 뒤 예상 위치
        self.last_vel = vel
        self.last_pred = pred
        # 화면 y는 아래로 갈수록 커짐. 예상 위치가 물고기보다 아래(> target)면 눌러 올림.
        self.last = pred > target
        return self.last


# ---------------------------------------------------------------- 정책 (IO 없는 상태기계, 테스트 가능)

class FishingBrain:
    """낚시 상태기계. IO 없이 관측값 -> 동작만 결정한다.

    상태: cast(던지기) -> wait(입질대기) -> fight(미니게임) -> reward(보상) -> cast ...
    step()이 반환하는 동작 문자열:
      'cast'    : 낚싯대를 던져라
      'hook'    : 입질! 후킹 클릭해라
      'recast'  : 입질 대기 시간 초과, 다시 던져라
      'press'   : 미니게임 - 마우스 누르고 유지(바 올림)
      'release' : 미니게임 - 마우스 떼기(바 내림)
      'hold'    : 현재 마우스 상태 유지
      'click'   : 보상창 닫기 클릭
      'none'    : 아무것도 안 함
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.reset()

    def reset(self):
        self.state = "cast"
        self.t = 0.0            # 현재 상태 진입 시각
        self.grace = 0.0        # 후킹 후 미니게임 대기 마감 시각
        self.ctrl = BarController(self.cfg)

    def step(self, now, bar, fish, bite):
        cfg = self.cfg
        in_game = bar is not None

        # 미니게임 바가 보이면 어느 상태든 즉시 fight 로 전환
        if in_game and self.state in ("cast", "wait"):
            self.state = "fight"
            self.t = now

        s = self.state

        if s == "cast":
            self.state = "wait"
            self.t = now
            return "cast"

        if s == "wait":
            if bite:
                self.state = "fight"
                self.t = now
                self.grace = now + float(cfg["minigame_grace"])
                return "hook"
            if now - self.t > float(cfg["wait_timeout"]):
                self.state = "cast"
                return "recast"
            return "none"

        if s == "fight":
            if in_game:
                # 바가 보이는 동안 grace 를 계속 갱신한다. 이렇게 하면 미니게임 중
                # 물고기가 바를 가리거나 검출이 순간 끊겨도(in_game=False) grace 안이라
                # 미니게임을 유지하고, 진짜로 끝났을 때만(끊김이 grace 초 지속) reward 로 간다.
                self.grace = now + float(cfg["minigame_grace"])
                self.t = now
                return "press" if self.ctrl.update(now, bar, fish) else "release"
            # 바 안 보임: 후킹 직후 미니게임 등장 대기 or 순간 검출 끊김 → grace 동안 현상태 유지
            if now < self.grace:
                return "none"
            self.state = "reward"
            self.t = now
            return "release"

        if s == "reward":
            if now - self.t > float(cfg["post_catch_delay"]):
                self.state = "cast"
                self.t = now
                return "click"
            return "none"

        return "none"


# ---------------------------------------------------------------- 캘리브레이션 (CLI)

def _select_roi_fullscreen(title, hint, reference_roi=None):
    with mss.mss() as sct:
        mon = sct.monitors[1]
        if reference_roi:
            x = reference_roi["left"] + reference_roi["width"] / 2
            y = reference_roi["top"] + reference_roi["height"] / 2
            mon = next((m for m in sct.monitors[1:] if m["left"] <= x < m["left"]+m["width"]
                        and m["top"] <= y < m["top"]+m["height"]), mon)
        img = np.array(sct.grab(mon))[:, :, :3].copy()
    print(hint)
    scale = min(1.0, 1400 / img.shape[1])
    disp = cv2.resize(img, None, fx=scale, fy=scale)
    r = cv2.selectROI(title, disp, showCrosshair=True)
    cv2.destroyAllWindows()
    if r[2] == 0 or r[3] == 0:
        return None
    return {
        "left": int(r[0] / scale) + mon["left"],
        "top": int(r[1] / scale) + mon["top"],
        "width": int(r[2] / scale),
        "height": int(r[3] / scale),
    }


def mode_calibrate():
    print("\n[캘리브레이션] 낚시 미니게임(세로 게이지)을 화면에 띄워두세요. 5초 후 캡처합니다.")
    for i in range(5, 0, -1):
        print(f"  {i}...")
        time.sleep(1)
    roi = _select_roi_fullscreen(
        "Select Fishing Track (Enter=OK)",
        "\n초록 바가 오르내리는 '세로 트랙'만 드래그하세요. (오른쪽 노란 진행바는 제외!)")
    if roi is None:
        print("취소되었습니다.")
        return
    cfg = load_config()
    if cfg.get("roi") != roi:
        cfg.pop("progress_roi", None)
    cfg["roi"] = roi
    save_config(cfg)
    print(f"트랙 영역 저장: {roi}")


def mode_calibrate_progress():
    cfg = load_config()
    print("5 秒后截图：切回游戏，显示钓鱼小游戏。请选择右侧完整进度槽内部，包含空白顶部，不含木框。")
    for i in range(5, 0, -1):
        print(f"  {i}...", flush=True)
        time.sleep(1)
    roi = _select_roi_fullscreen(
        "Select FULL Progress Slot - exclude border (Enter=OK, Esc=Cancel)",
        "请选择完整进度槽，不要只选当前有颜色的部分。", reference_roi=cfg.get("roi"))
    if roi is None:
        print("已取消，原选区保持不变。")
        return
    if roi["width"] < 2 or roi["height"] < 30:
        raise ValueError("进度槽选区太小，请重新选择。")
    cfg = load_config()
    cfg["progress_roi"] = roi
    save_config(cfg)
    print(f"进度槽已保存：{roi}")


def mode_calibrate_bite():
    print("\n[입질 감지영역 캘리브레이션] 낚싯줄을 던진 상태(찌가 물에 떠 있는 상태)로 두세요. 5초 후 캡처.")
    for i in range(5, 0, -1):
        print(f"  {i}...")
        time.sleep(1)
    roi = _select_roi_fullscreen(
        "Select Bite '!' Region (Enter=OK)",
        "\n찌 바로 위쪽 물 영역을 드래그하세요. (입질 시 '!' 말풍선이 뜨는 자리)")
    if roi is None:
        print("취소되었습니다.")
        return
    cfg = load_config()
    cfg["bite_roi"] = roi
    save_config(cfg)
    print(f"입질 감지영역 저장: {roi}")


# ---------------------------------------------------------------- 색상 튜닝

def mode_tune():
    cfg = load_config()
    print("\n[튜닝] 미니게임을 띄워두세요. q=종료. 왼쪽=원본 / 가운데=초록바 / 오른쪽=물고기")
    with mss.mss() as sct:
        while True:
            frame = grab(sct, cfg["roi"])
            bar, fish, bar_mask, fish_mask = detect(frame, cfg)
            view = np.hstack([
                frame,
                cv2.cvtColor(bar_mask, cv2.COLOR_GRAY2BGR),
                cv2.cvtColor(fish_mask, cv2.COLOR_GRAY2BGR),
            ])
            cv2.imshow("tune (q=quit)", view)
            if cv2.waitKey(30) & 0xFF == ord("q"):
                break
    cv2.destroyAllWindows()


# ---------------------------------------------------------------- 봇 본체 (CLI)

class Bot:
    def __init__(self):
        self.cfg = load_config()
        self.enabled = True
        self.quit = False
        self.mouse = Mouse()
        self.brain = FishingBrain(self.cfg)
        self.ctrl = BarController(self.cfg)
        self.full_auto = bool(self.cfg.get("full_auto", False))
        self._bite_base = None
        self._bite_ready_t = 0.0
        self.audio = None
        if self.cfg.get("bite_mode", "sound") == "sound":
            from audio_bite import AudioBiteDetector
            self.audio = AudioBiteDetector(self.cfg, log=lambda m: print(f"[오디오] {m}"))
        self._hwnd = None

    def _ensure_focus(self):
        """게임 창이 포그라운드가 아니면 포커스를 가져온다(입력 전달 보장)."""
        if not self.cfg.get("auto_focus", True):
            return
        if self._hwnd is None:
            self._hwnd = find_game_hwnd(self.cfg.get("game_title", "Stardew"))
        if self._hwnd and ctypes.windll.user32.GetForegroundWindow() != self._hwnd:
            focus_window(self._hwnd)

    def on_key(self, key):
        if key == keyboard.Key.f8:
            self.enabled = not self.enabled
            print(f"[F8] 봇 {'재개' if self.enabled else '일시정지'}")
            if not self.enabled:
                self.mouse.up()
        elif key == keyboard.Key.f9:
            self.full_auto = not self.full_auto
            self.brain.reset()
            print(f"[F9] 완전자동 {'ON' if self.full_auto else 'OFF'}")
        elif key == keyboard.Key.esc:
            self.quit = True
            return False

    def _do_cast(self):
        cfg = self.cfg
        tgt = cfg.get("cast_target")
        if tgt:
            self.mouse.move(tgt["x"], tgt["y"])
            time.sleep(0.05)
        self.mouse.click(hold=float(cfg["cast_power"]))
        self._bite_base = None
        self._bite_ready_t = time.time() + float(cfg["cast_settle"])

    def _read_bite(self, sct, now):
        """입질 여부(bool). cast_settle 이후부터 감시 시작."""
        if now < self._bite_ready_t:
            return False
        cur = grab(sct, self.cfg["bite_roi"])
        if self._bite_base is None:
            self._bite_base = cur
            return False
        frac = bite_change_fraction(self._bite_base, cur)
        return frac > float(self.cfg["bite_change"])

    def run(self):
        cfg = self.cfg
        interval = 1.0 / max(int(cfg["fps"]), 5)

        listener = keyboard.Listener(on_press=self.on_key)
        listener.start()

        print("\n[봇 실행 중] F8: 일시정지/재개  F9: 완전자동 토글  ESC: 종료")
        print(f"현재 완전자동: {'ON' if self.full_auto else 'OFF'}")
        print("완전자동 OFF일 때는 직접 던지고 후킹하면, 미니게임만 봇이 잡습니다.")

        with mss.mss() as sct:
            while not self.quit:
                t0 = time.time()
                if not self.enabled:
                    time.sleep(0.1)
                    continue

                frame = grab(sct, cfg["roi"])
                bar, fish, _, _ = detect(frame, cfg)

                # 입력이 필요한 상황이면 게임 창을 포그라운드로
                if self.full_auto or bar is not None:
                    self._ensure_focus()

                if self.full_auto:
                    bite = False
                    if self.brain.state == "wait":
                        if self.audio is not None:
                            if t0 >= self._bite_ready_t:
                                self.audio.arm(True)
                                bite = self.audio.consume_spike()
                        else:
                            bite = self._read_bite(sct, t0)
                    elif self.audio is not None:
                        self.audio.arm(False)
                    action = self.brain.step(t0, bar, fish, bite)
                    self._apply(action, sct)
                else:
                    # 미니게임 보조 전용 모드 (예측 제어 + 최소 펄스)
                    if bar is not None:
                        press = self.ctrl.update(t0, bar, fish)
                        self.mouse.set(press, t0, float(cfg.get("min_pulse", 0.0)))
                    else:
                        self.ctrl.reset()
                        self.mouse.up()

                elapsed = time.time() - t0
                if elapsed < interval:
                    time.sleep(interval - elapsed)

        self.mouse.up()
        listener.stop()
        print("종료했습니다.")

    def _apply(self, action, sct):
        if action == "cast":
            print("  >> 캐스팅")
            self._do_cast()
        elif action == "hook":
            print("  >> 입질! 후킹")
            self.mouse.click()
        elif action == "recast":
            print("  >> 입질 없음, 재캐스팅")
        elif action == "press":
            self.mouse.down()
        elif action == "release":
            self.mouse.up()
        elif action == "click":
            print("  >> 보상 확인, 다음 낚시")
            self.mouse.click()
        # 'hold', 'none' : 아무것도 안 함


# ---------------------------------------------------------------- 엔트리

if __name__ == "__main__":
    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass

    pydirectinput.PAUSE = 0
    pydirectinput.FAILSAFE = False

    mode = sys.argv[1] if len(sys.argv) > 1 else "run"
    if mode == "calibrate":
        mode_calibrate()
    elif mode == "calibrate-bite":
        mode_calibrate_bite()
    elif mode == "calibrate-progress":
        mode_calibrate_progress()
    elif mode == "tune":
        mode_tune()
    elif mode == "run":
        Bot().run()
    else:
        print("사용법: python main.py [calibrate|calibrate-bite|tune|run]")
