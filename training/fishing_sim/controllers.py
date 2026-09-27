"""Baselines only consume deployment-shaped observations, never simulator truth."""
import numpy as np
from .core import FPS, FISH_CENTER_OFFSET, LANE_HEIGHT
from .observations import latest_geometry


class PredictiveRule:
    def __call__(self, observation):
        g = latest_geometry(observation)
        error = g["fish_center"] + 0.12 * g["fish_velocity"] - g["bar_center"] - 0.12 * g["bar_velocity"]
        if abs(error) < 0.02 * LANE_HEIGHT:
            return g["held"]
        return int(error < 0)


class MPC:
    def __init__(self, horizon=8):
        self.horizon = horizon
        self.actions = ((np.arange(2**horizon)[:, None] >> np.arange(horizon)) & 1).astype(bool)

    def __call__(self, observation):
        g = latest_geometry(observation)
        height, tackle = float(g["bar_height"]), g["tackle"]
        y = np.full(len(self.actions), g["bar_center"] - height / 2, dtype=np.float64)
        velocity = np.full_like(y, g["bar_velocity"] / FPS)
        # Predict with measured displacement, not the engine's fish target/speed.
        fish = float(g["fish_center"] - FISH_CENTER_OFFSET)
        fv = float(g["fish_velocity"] / FPS)
        cost = np.zeros_like(y)
        previous = np.full(len(y), bool(g["held"]))
        for k in range(self.horizon):
            held = self.actions[:, k]
            cost += 0.0001 * (held != previous)
            previous = held
            for _ in range(2):
                fish = np.clip(fish + fv, 0, 269)
                inside = ((fish + 6 <= y - 16 + height) & (fish - 8 >= y - 16)) | ((fish >= 274 - height) & (y >= 280 - height))
                velocity = np.where(held & ((y <= 6.00001) | (y >= 288 - height - 0.00001)), 0, velocity)
                if tackle == "barbed":
                    velocity += np.where(inside, np.where(fish + 8 < y + height / 2, -0.2, 0.2), 0)
                gravity = np.where(held, -0.125, 0.125) * np.where(inside, 0.3 if tackle == "barbed" else 0.6, 1)
                velocity += gravity
                y += velocity
                bottom, top = y > 288 - height, y < 6
                velocity = np.where(bottom, -velocity * (2 / 3) * (0.1 if tackle == "lead" else 1), velocity)
                velocity = np.where(top, -velocity * (2 / 3), velocity)
                y = np.clip(y, 6, 288 - height)
                # Hitbox margin + a small momentum penalty to anticipate overshoot.
                dist = np.maximum(np.maximum(y + 9 - (fish + 15), (fish + 15) - (y + height - 9)), 0) / LANE_HEIGHT
                cost += dist * dist + 0.001 * ((velocity - fv) * FPS / LANE_HEIGHT) ** 2
        return int(self.actions[int(np.argmin(cost)), 0])
