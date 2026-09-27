"""Causal sound-effect fingerprint matching; no audio device or input control."""
from collections import deque
from pathlib import Path

import numpy as np

SIGNATURE = Path(__file__).resolve().parent / "auto_assets" / "bite_signature.npz"


class BiteSignatureMatcher:
    """Match a short time-frequency pattern after adaptive spectral subtraction.

    Scores are cosine similarities, not calibrated probabilities. Two adjacent
    windows must support a match, with one above the stronger threshold.
    """
    def __init__(self, path=SIGNATURE):
        with np.load(path, allow_pickle=False) as saved:
            schema = str(saved["schema"])
            self.rate = int(saved["sample_rate"])
            self.nfft = int(saved["nfft"])
            self.hop = int(saved["hop"])
            self.bins = tuple(map(int, saved["bins"]))
            self.template = saved["template"].astype(np.float32)
            self.threshold = float(saved["threshold"])
            self.support_threshold = float(saved["support_threshold"])
        if (schema != "stardew-bite-spectrum-v1" or self.rate != 24000 or self.nfft != 1024
                or self.hop != 256 or self.template.ndim != 2
                or self.template.shape[0] != self.bins[1]-self.bins[0]
                or not 0 < self.support_threshold <= self.threshold <= 1
                or not np.isfinite(self.template).all() or float(np.linalg.norm(self.template)) < 1e-8):
            raise ValueError("Invalid bite audio signature")
        self.template /= max(float(np.linalg.norm(self.template)), 1e-10)
        self.frames = self.template.shape[1]
        self.window_seconds = (self.nfft+(self.frames-1)*self.hop)/self.rate
        self.window = np.hanning(self.nfft).astype(np.float32)
        # 48 -> 24 kHz with an anti-alias filter; retain history across blocks.
        taps = np.arange(31, dtype=np.float32)-15
        self.fir = np.sinc(taps*(20000/48000))*np.hamming(31)
        self.fir = (self.fir/self.fir.sum()).astype(np.float32)
        self.reset()

    def reset(self):
        self.pending = np.empty(0, np.float32)
        self.filter_history = np.zeros(len(self.fir)-1, np.float32)
        self.decimation_phase = 0
        self.noise = np.zeros(self.template.shape[0], np.float32)
        self.history = deque(maxlen=self.frames)
        self.previous_score = 0.
        self.score = self.support = 0.
        self.processed_samples = 0

    def feed(self, audio, sample_rate=48000):
        """Return matched windows ending inside this block, in sample time."""
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim == 2:
            combined = audio.mean(1)
            channel_energy = np.mean(audio*audio, axis=0)
            if np.mean(combined*combined) < float(channel_energy.max())*.0625:
                combined = audio[:, int(channel_energy.argmax())]
            audio = combined
        if audio.ndim != 1 or not np.isfinite(audio).all():
            self.reset()
            return []
        if sample_rate == 48000:
            joined = np.concatenate((self.filter_history, audio))
            self.filter_history = joined[-len(self.filter_history):].copy()
            filtered = np.convolve(joined, self.fir, mode="valid")
            start = self.decimation_phase
            self.decimation_phase = (start-len(filtered)) % 2
            audio = filtered[start::2]
        elif sample_rate != self.rate:
            raise ValueError("Audio fingerprint expects 24 or 48 kHz")
        self.pending = np.concatenate((self.pending, audio))
        matches = []
        consumed = 0
        while len(self.pending)-consumed >= self.nfft:
            samples = self.pending[consumed:consumed+self.nfft]
            spectrum = np.abs(np.fft.rfft(samples*self.window))[slice(*self.bins)]/(self.nfft/2)
            feature = np.sqrt(np.maximum(spectrum-.9*self.noise, 0))
            self.noise = .99*self.noise+.01*spectrum
            self.history.append(feature)
            if len(self.history) == self.frames:
                pattern = np.stack(self.history, axis=1)
                self.score = float(np.sum(pattern*self.template)/max(float(np.linalg.norm(pattern)), 1e-10))
                self.support = min(self.previous_score, self.score)
                peak = max(self.previous_score, self.score)
                if (self.support >= self.support_threshold and peak >= self.threshold
                        and float(np.mean(samples*samples)) >= 4e-8):
                    matches.append({"sample_time": (self.processed_samples+self.nfft)/self.rate,
                                    "score": peak, "support": self.support})
                self.previous_score = self.score
            self.processed_samples += self.hop
            consumed += self.hop
        self.pending = self.pending[consumed:].copy()
        return matches
