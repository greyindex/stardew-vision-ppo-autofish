"""Fixed 60 Hz numerical model of Pufferdle's catch-only minigame.

Coordinate/update conventions deliberately follow upstream revision 506322f.
See SOURCES.md for the reference, intentional changes, and known limitations.
There are no rendering, OS-input, clock, or screen-capture dependencies here.
"""
from dataclasses import dataclass
import json
import math
from pathlib import Path
import random

FPS = 60
LANE_TOP = 6.0
LANE_BOTTOM = 288.0
LANE_HEIGHT = LANE_BOTTOM - LANE_TOP
FISH_CENTER_OFFSET = 15.0  # center of upstream's [fishPos+8, fishPos+22] hitbox
BEHAVIORS = ("Mixed", "Dart", "Smooth", "Sinker", "Floater")
TACKLES = ("none", "cork", "lead", "trap", "barbed")
CATALOG = json.loads(Path(__file__).with_name("fish.json").read_text(encoding="utf-8"))
BY_NAME = {fish["name"]: fish for fish in CATALOG}


@dataclass(frozen=True)
class PhysicsConfig:
    level: int = 5
    tackle: str = "none"
    max_seconds: float = 120.0

    def __post_init__(self):
        if not 0 <= self.level <= 20:
            raise ValueError("level must be between 0 and 20")
        if self.tackle not in TACKLES:
            raise ValueError(f"Unsupported tackle: {self.tackle}")
        if not math.isfinite(self.max_seconds) or self.max_seconds <= 0:
            raise ValueError("max_seconds must be positive and finite")


def in_bar(fish_pos, bar_top, height):
    """Upstream's hit test, including special containment at the bottom."""
    return ((fish_pos + 6 <= bar_top - 16 + height)
            and (fish_pos - 8 >= bar_top - 16)) or (
        fish_pos >= 274 - height and bar_top >= 280 - height)


def advance_bar(bar_top, velocity, height, held, contained, tackle="none", fish_pos=None):
    gravity = -0.125 if held else 0.125
    if held and (bar_top == LANE_TOP or bar_top == LANE_BOTTOM - height):
        velocity = 0.0
    if contained:
        gravity *= 0.3 if tackle == "barbed" else 0.6
        if tackle == "barbed" and fish_pos is not None:
            velocity += -0.2 if fish_pos + 8 < bar_top + height / 2 else 0.2
    velocity += gravity
    bar_top += velocity
    if bar_top + height > LANE_BOTTOM:
        bar_top = LANE_BOTTOM - height
        velocity *= -(2.0 / 3.0) * (0.1 if tackle == "lead" else 1.0)
    elif bar_top < LANE_TOP:
        bar_top = LANE_TOP
        velocity *= -(2.0 / 3.0)
    return bar_top, velocity


class FishingPhysics:
    def __init__(self, fish, config=None, seed=0, rng=None):
        self.fish = dict(BY_NAME[fish] if isinstance(fish, str) else fish)
        self.config = config or PhysicsConfig()
        self.difficulty = float(self.fish["difficulty"])
        self.behavior = self.fish["behavior"]
        if self.behavior not in BEHAVIORS or not 0 <= self.difficulty <= 120:
            raise ValueError("Invalid fish mechanics")
        self.rng = rng if rng is not None else random.Random(int(seed))
        self.height = 48.0 + 4 * self.config.level + (12 if self.config.tackle == "cork" else 0)
        self.bar_top = LANE_BOTTOM - self.height
        self.bar_speed = 0.0
        self.fish_pos = 254.0
        self.fish_target = (100.0 - self.difficulty) / 100.0 * 274.0
        self.fish_speed = 0.0
        self.progress = 0.3
        self.perfect = True
        self.contained = True
        self.ticks = 0
        self.inside_ticks = 0
        self.done = False
        self.outcome = "running"

    def rand_range(self, low, high):
        # JS uses floor(random * (high-low)) + low, even for fractional bounds.
        return math.floor(self.rng.random() * (high - low)) + low

    def advance_fish(self):
        d, kind = self.difficulty, self.behavior
        smooth = kind == "Smooth"
        if self.rng.random() < d * (20 if smooth else 1) / 4000 and (
                not smooth or self.fish_target == -1):
            below, above = 274 - self.fish_pos, self.fish_pos
            fraction = min(99, d + self.rand_range(10, 45)) / 100
            self.fish_target = self.fish_pos + self.rand_range(min(-above, below), below) * fraction
        # Upstream resets this local each tick. Preserve that behavior, not an
        # unverified persistent acceleration inferred from variable names.
        bias = -0.01 if kind == "Floater" else 0.01 if kind == "Sinker" else 0.0
        if abs(self.fish_pos - self.fish_target) > 3 and self.fish_target != -1:
            accel = (self.fish_target - self.fish_pos) / (
                self.rand_range(10, 30) + 100 - min(100, d))
            self.fish_speed += (accel - self.fish_speed) / 5
        elif not smooth and self.rng.random() < d / 2000:
            jump = self.rand_range(-100, 51) if self.rng.random() < 0.5 else self.rand_range(50, 101)
            self.fish_target = self.fish_pos + jump
        else:
            self.fish_target = -1.0
        if kind == "Dart" and self.rng.random() < d / 1000:
            jump = self.rand_range(-100 - d * 2, -51) if self.rng.random() < 0.5 else self.rand_range(50, 101 + d * 2)
            self.fish_target = self.fish_pos + jump
        self.fish_target = max(-1.0, min(self.fish_target, 274.0))
        self.fish_pos = max(0.0, min(self.fish_pos + self.fish_speed + bias, 269.0))

    def step(self, held):
        if self.done:
            raise RuntimeError("Reset before stepping a finished episode")
        if held not in (0, 1, False, True):
            raise ValueError("The only actions are release=0 and hold=1")
        self.advance_fish()
        # Membership is sampled after fish motion and before bar motion upstream.
        self.contained = in_bar(self.fish_pos, self.bar_top, self.height)
        self.bar_top, self.bar_speed = advance_bar(
            self.bar_top, self.bar_speed, self.height, held, self.contained, self.config.tackle, self.fish_pos)
        self.progress += 0.002 if self.contained else (-0.002 if self.config.tackle == "trap" else -0.003)
        self.progress = min(1.0, max(0.0, self.progress))
        self.perfect = self.perfect and self.contained
        self.ticks += 1
        self.inside_ticks += int(self.contained)
        if self.progress >= 1:
            self.done, self.outcome = True, "caught"
        elif self.progress <= 0:
            self.done, self.outcome = True, "lost"
        elif self.ticks >= math.ceil(self.config.max_seconds * FPS):
            self.done, self.outcome = True, "timeout"

    def measurement(self):
        """Only quantities a future visual adapter can estimate; no hidden state."""
        return {
            "timestamp": self.ticks / FPS,
            "fish_center": self.fish_pos + FISH_CENTER_OFFSET,
            "bar_top": self.bar_top,
            "bar_bottom": self.bar_top + self.height,
            "progress": self.progress,
        }

    def diagnostics(self):
        return {
            "fish": self.fish["name"], "difficulty": self.difficulty,
            "behavior": self.behavior, "level": self.config.level,
            "tackle": self.config.tackle, "outcome": self.outcome,
            "caught": self.outcome == "caught", "perfect": self.perfect and self.outcome == "caught",
            "seconds": self.ticks / FPS,
            "inside_fraction": self.inside_ticks / max(self.ticks, 1),
        }
