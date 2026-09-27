"""Shared observation contract for simulated geometry and future visual measurements."""
import numpy as np
from .core import LANE_TOP, LANE_HEIGHT, TACKLES

SCHEMA = "geometry-v1"
HISTORY = 16
FRAME_FIELDS = (
    "fish_y", "bar_y", "bar_height", "fish_v_est", "bar_v_est", "progress",
    "progress_delta", "button_commanded", "since_edge", "frame_dt",
    "fish_seen", "bar_seen", "progress_seen", "image_age", "fish_vel_valid",
    "bar_vel_valid", "history_valid", "time_left",
)
CONFIG_FIELDS = ("level",) + tuple("tackle_" + name for name in TACKLES)
FRAME_SIZE = len(FRAME_FIELDS)
OBS_SIZE = HISTORY * FRAME_SIZE + len(CONFIG_FIELDS)


class ObservationEncoder:
    def __init__(self, config):
        self.config = config
        self.history = np.zeros((HISTORY, FRAME_SIZE), dtype=np.float32)
        self.config_vector = np.zeros(len(CONFIG_FIELDS), dtype=np.float32)
        self.config_vector[0] = config.level / 20.0
        self.config_vector[1 + TACKLES.index(config.tackle)] = 1.0
        self.previous = None

    def encode(self, measurement, commanded, since_edge):
        m, p = measurement, self.previous
        center = (m["bar_top"] + m["bar_bottom"]) / 2
        dt = m["timestamp"] - p["timestamp"] if p is not None else 0.0
        velocity_valid = dt > 0
        fv = (m["fish_center"] - p["fish_center"]) / dt / LANE_HEIGHT / 3 if velocity_valid else 0.0
        bv = (center - (p["bar_top"] + p["bar_bottom"]) / 2) / dt / LANE_HEIGHT / 3 if velocity_valid else 0.0
        delta = m["progress"] - p["progress"] if p is not None else 0.0
        row = [
            (m["fish_center"] - LANE_TOP) / LANE_HEIGHT,
            (center - LANE_TOP) / LANE_HEIGHT,
            (m["bar_bottom"] - m["bar_top"]) / LANE_HEIGHT,
            fv, bv, m["progress"], delta / 0.02, float(commanded),
            since_edge / 0.5, dt / 0.1,
            1, 1, 1, 0, float(velocity_valid), float(velocity_valid), 1,
            max(0.0, 1 - m["timestamp"] / self.config.max_seconds),
        ]
        self.history[:-1] = self.history[1:]
        self.history[-1] = row
        np.clip(self.history[-1], -1.0, 1.0, out=self.history[-1])
        self.previous = dict(m)
        return np.concatenate((self.history.ravel(), self.config_vector))


def latest_geometry(observation):
    """Used by both baselines: only decode the same input supplied to PPO."""
    obs = np.asarray(observation)
    row = obs[HISTORY * FRAME_SIZE - FRAME_SIZE:HISTORY * FRAME_SIZE]
    return {
        "fish_center": row[0] * LANE_HEIGHT + LANE_TOP,
        "bar_center": row[1] * LANE_HEIGHT + LANE_TOP,
        "bar_height": row[2] * LANE_HEIGHT,
        "fish_velocity": row[3] * LANE_HEIGHT * 3,
        "bar_velocity": row[4] * LANE_HEIGHT * 3,
        "held": int(row[7] > 0.5),
        "tackle": TACKLES[int(np.argmax(obs[HISTORY * FRAME_SIZE + 1:]))],
    }
