"""Gymnasium environment: geometry observations, binary held-state actions."""
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from .core import CATALOG, BY_NAME, FPS, FishingPhysics, PhysicsConfig
from .observations import ObservationEncoder, OBS_SIZE


class FishingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, stage="full", fish=None, level=None, tackle="none", max_seconds=120.0):
        super().__init__()
        if stage not in ("easy", "mixed", "full"):
            raise ValueError("stage must be easy, mixed or full")
        self.stage, self.fixed_fish, self.fixed_level = stage, fish, level
        self.tackle, self.max_seconds = tackle, max_seconds
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(-1.0, 1.0, (OBS_SIZE,), np.float32)
        self.physics = None

    def set_stage(self, stage):
        if stage not in ("easy", "mixed", "full"):
            raise ValueError(stage)
        self.stage = stage  # Existing fights finish with their current config.

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        name = options.get("fish", self.fixed_fish)
        if name is None:
            maximum = 55 if self.stage == "easy" else 85 if self.stage == "mixed" else 120
            pool = [fish for fish in CATALOG if fish["difficulty"] <= maximum]
            fish = pool[int(self.np_random.integers(len(pool)))]
        else:
            if name not in BY_NAME:
                raise ValueError(f"Unknown fish: {name}")
            fish = BY_NAME[name]
        level = options.get("level", self.fixed_level)
        if level is None:
            level = 10 if self.stage == "easy" else int(self.np_random.integers(5 if self.stage == "mixed" else 0, 11))
        config = PhysicsConfig(level=int(level), tackle=options.get("tackle", self.tackle), max_seconds=self.max_seconds)
        self.physics = FishingPhysics(fish, config, seed=int(self.np_random.integers(2**63)))
        self.encoder = ObservationEncoder(config)
        self.commanded, self.last_edge, self.switches = 0, 0.0, 0
        self.observation = self.encoder.encode(self.physics.measurement(), 0, 0)
        return self.observation.copy(), self.physics.diagnostics()

    def step(self, action):
        if self.physics is None or self.physics.done:
            raise RuntimeError("Call reset before step")
        if not self.action_space.contains(action):
            raise ValueError("Expected action 0 (release) or 1 (hold)")
        action = int(action)
        switched = action != self.commanded
        if switched:
            self.last_edge = self.physics.ticks / FPS
            self.switches += 1
        self.commanded = action
        old_phi = 0.5 * self.physics.progress
        old_ticks = self.physics.ticks
        for _ in range(2):  # Fixed 60 Hz physics, 30 Hz controller.
            self.physics.step(action)
            if self.physics.done:
                break
        terminal = self.physics.done
        phi = 0.0 if terminal else 0.5 * self.physics.progress
        outcome_reward = (1.0 if self.physics.outcome == "caught" else -1.0) if terminal else 0.0
        reward = outcome_reward + phi - old_phi - 0.001 * (self.physics.ticks - old_ticks) / FPS - 0.0001 * switched
        self.observation = self.encoder.encode(
            self.physics.measurement(), action, self.physics.ticks / FPS - self.last_edge)
        info = {}
        if terminal:
            info = self.physics.diagnostics()
            info["switches_per_second"] = self.switches / max(info["seconds"], 1 / FPS)
            info["is_success"] = info["caught"]
        # A configured deadline is an observable task failure, not a collection cutoff.
        return self.observation.copy(), float(reward), terminal, False, info
