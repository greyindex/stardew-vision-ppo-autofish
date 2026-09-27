"""
입질 소리 감지 (WASAPI 루프백)
================================
스피커로 나가는 게임 소리를 그대로 캡처해서, 조용한 대기 상태에서 갑자기
튀는 소리(입질음 "똑")를 순간 에너지 스파이크로 감지한다.

- 배경 스레드가 계속 짧은 블록을 녹음하며 RMS(음량)와 완만한 배경 기준선을 갱신
- arm()으로 켠 동안만 스파이크를 기록 (캐스팅/미니게임 소리 오탐 방지)
- consume_spike()로 스파이크 발생 여부를 1회성으로 가져감
"""

import time
import threading
import ctypes
import sys

import numpy as np


def prepare_audio_backend(log=lambda message: None):
    """Import on the persistent owner thread before starting recorder threads.

    SoundCard 0.4.6 initializes COM only on its first importing thread. Each
    later recorder thread must also initialize COM itself (see _run).
    """
    try:
        import soundcard  # noqa: F401
        return True
    except Exception as exc:
        log(f"声音辅助不可用，将继续使用视觉：{exc}")
        return False


def is_spike(rms, baseline, floor, ratio):
    """순간 음량이 절대 바닥값과 배경 기준선 배수를 동시에 넘으면 스파이크."""
    return rms > floor and rms > baseline * ratio


class AudioBiteDetector:
    def __init__(self, cfg, log=lambda m: None):
        self.cfg = cfg
        self.log = log
        self.level = 0.0      # 최근 블록 RMS (UI 표시용)
        self.baseline = 0.0   # 완만한 배경 기준선
        self.ok = False       # 캡처 정상 여부
        self._spike = False
        self._event = None
        self._last_spike_t = 0.0
        self._armed = False
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self.thread = threading.Thread(target=self._run, name="bite-loopback", daemon=True)
        self.thread.start()

    def arm(self, on=True):
        """Clear a stale spike only when listening starts, not on every frame."""
        with self._lock:
            if not on or not self._armed:
                self._spike = False
                self._event = None
            self._armed = on

    def consume_spike(self):
        return self.consume_event() is not None

    def consume_event(self, max_age=.30):
        """Return a fresh audio cue, never an old queued sound from another phase."""
        with self._lock:
            event = self._event
            self._spike = False
            self._event = None
        if event and 0 <= time.perf_counter() - event["time"] <= max_age:
            return event
        return None

    def snapshot(self):
        with self._lock:
            return {"ok": self.ok, "rms": self.level, "baseline": self.baseline,
                    "armed": self._armed}

    def stop(self):
        self.arm(False)
        self._stop.set()

    def _report(self, message):
        try:
            self.log(message)
        except Exception:
            pass

    def _run(self):
        ole32 = None
        com_owned = False
        try:
            # Import first: SoundCard's own first-import initialization rejects
            # S_FALSE. Our explicit per-thread initialization accepts 0 and 1.
            import soundcard as sc
            if sys.platform == "win32":
                ole32 = ctypes.OleDLL("ole32")
                ole32.CoInitializeEx.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
                ole32.CoInitializeEx.restype = ctypes.c_long
                ole32.CoUninitialize.argtypes = []
                ole32.CoUninitialize.restype = None
                result = ole32.CoInitializeEx(None, 0)  # COINIT_MULTITHREADED
                if result not in (0, 1):
                    raise OSError(f"CoInitializeEx: 0x{result & 0xffffffff:08x}")
                com_owned = True
        except Exception as e:
            self._report(f"声音辅助初始化失败，继续视觉检测：{e}")
            return

        sr, block = 48000, 1024
        try:
            while not self._stop.is_set():
                try:
                    spk = sc.default_speaker()
                    if spk is None:
                        raise RuntimeError("没有默认音频输出设备")
                    mic = sc.get_microphone(id=str(spk.name), include_loopback=True)
                    # WASAPI's single-channel capture can return corrupt data on
                    # some devices. Capture all channels and combine their energy.
                    with mic.recorder(samplerate=sr, blocksize=block) as rec:
                        self._report(f"声音辅助已连接：{spk.name}（需要视觉确认）")
                        ema = None
                        previous = time.perf_counter()
                        warm_until = previous + .30
                        while not self._stop.is_set():
                            data = rec.record(numframes=block//2)
                            now = time.perf_counter()
                            gap = now - previous
                            previous = now
                            if not data.size or not np.isfinite(data).all() or gap > .15:
                                self.ok = False
                                ema = None
                                warm_until = now + .30
                                with self._lock:
                                    self._spike, self._event = False, None
                                continue
                            rms = float(np.sqrt(np.mean(np.square(data.astype(np.float64)))) + 1e-9)
                            self.level = rms
                            if ema is None:
                                ema = max(rms, 1e-5)
                            # Background EMA; a transient cannot lift it sharply.
                            ema = .97 * ema + .03 * min(rms, ema * 1.5)
                            self.baseline = ema
                            self.ok = now >= warm_until
                            ratio = float(self.cfg.get("bite_sound_ratio", 3.5))
                            floor = float(self.cfg.get("bite_sound_floor", 0.03))
                            refr = float(self.cfg.get("bite_refractory", 1.0))
                            with self._lock:
                                if (self.ok and self._armed and is_spike(rms, ema, floor, ratio)
                                        and now - self._last_spike_t > refr):
                                    self._last_spike_t = now
                                    self._spike = True
                                    self._event = {"time": now, "rms": rms, "baseline": ema}
                except Exception as e:
                    self.ok = False
                    with self._lock:
                        self._spike, self._event = False, None
                    self._report(f"声音辅助连接异常，1 秒后重连；视觉仍工作：{e}")
                    self._stop.wait(1.0)
        finally:
            self.ok = False
            if com_owned:
                ole32.CoUninitialize()
