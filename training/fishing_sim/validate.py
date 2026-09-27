"""Meaningful mechanics and observation-contract checks before spending training time."""
import hashlib
import json
from pathlib import Path
import random
import shutil
import subprocess
import unittest
import numpy as np
from .core import BEHAVIORS, CATALOG, TACKLES, FishingPhysics, PhysicsConfig
from .env import FishingEnv
from .observations import OBS_SIZE, ObservationEncoder

ROOT = Path(__file__).resolve().parent.parent


class MechanicsChecks(unittest.TestCase):
    def test_gymnasium_contract(self):
        from gymnasium.utils.env_checker import check_env
        from stable_baselines3.common.env_checker import check_env as sb3_check
        check_env(FishingEnv(), skip_render_check=True)
        sb3_check(FishingEnv())

    def test_seed_and_observation_contract(self):
        a, b = FishingEnv(), FishingEnv()
        oa, ia = a.reset(seed=90210)
        ob, ib = b.reset(seed=90210)
        self.assertEqual(ia, ib)
        self.assertEqual(oa.shape, (OBS_SIZE,))
        for action in [1, 0, 1, 1, 0] * 80:
            ra, rb = a.step(action), b.step(action)
            np.testing.assert_array_equal(ra[0], rb[0])
            self.assertEqual(ra[1:], rb[1:])
            self.assertTrue(a.observation_space.contains(ra[0]))
            if ra[2]:
                break
        before = a.physics.measurement()
        a.physics.fish_target = 123456
        a.physics.fish_speed = -98765
        self.assertEqual(before, a.physics.measurement())
        e1, e2 = ObservationEncoder(a.physics.config), ObservationEncoder(a.physics.config)
        np.testing.assert_array_equal(e1.encode(before, 0, 0), e2.encode(a.physics.measurement(), 0, 0))

    def test_action_controls_momentum_and_bounds(self):
        for tackle in TACKLES:
            p = FishingPhysics("Legend", PhysicsConfig(level=0, tackle=tackle), seed=52)
            initial = p.bar_top
            for _ in range(20):
                p.step(1)
            self.assertLess(p.bar_top, initial)
            for t in range(2000):
                if p.done:
                    break
                p.step((t // 45) % 2)
                self.assertGreaterEqual(p.bar_top, 6)
                self.assertLessEqual(p.bar_top + p.height, 288)
                self.assertGreaterEqual(p.fish_pos, 0)
                self.assertLessEqual(p.fish_pos, 269)

    def test_terminal_reward_and_deadline(self):
        env = FishingEnv(fish="Carp", level=10, max_seconds=1/60)
        env.reset(seed=10)
        _, reward, terminated, truncated, info = env.step(0)
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual(info["outcome"], "timeout")
        self.assertAlmostEqual(reward, -1.15 - 0.001/60)
        with self.assertRaises(RuntimeError):
            env.step(0)
        p = FishingPhysics("Carp", PhysicsConfig(level=10))
        p.advance_fish = lambda: None
        for _ in range(351):
            if p.done:
                break
            p.step(0)
        self.assertEqual(p.outcome, "caught")
        self.assertTrue(p.perfect)

    def test_catalog_coverage(self):
        self.assertEqual(len(CATALOG), 55)
        self.assertEqual({f["behavior"] for f in CATALOG}, set(BEHAVIORS))
        self.assertEqual(len({f["name"] for f in CATALOG}), len(CATALOG))

    def test_pinned_javascript_reference(self):
        source = ROOT / ".reference" / "FishingGame.js"
        if not source.exists() or not shutil.which("node"):
            self.skipTest("Run fetch_reference.ps1 and install Node for the optional source parity check")
        manifest = json.loads((ROOT / "source_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), manifest["files"]["FishingGame.js"]["sha256"])
        class Tape:
            def __init__(self, values):
                self.values = iter(values)
            def random(self):
                return next(self.values)
        cases, expected = [], []
        names = ["Legend", "Scorpion Carp", "Tuna", "Octopus", "Pufferfish"]
        upstream_tackle = {"none": "", "cork": "corkBobber", "lead": "leadBobber", "trap": "trapBobber", "barbed": "barbedHook"}
        for name in names:
            for tackle in TACKLES:
                generator = random.Random(1024 + len(cases))
                tape = [generator.random() for _ in range(4000)]
                actions = [(tick // 50) % 2 for tick in range(400)]
                p = FishingPhysics(name, PhysicsConfig(level=5, tackle=tackle), rng=Tape(tape))
                cases.append({"tape": tape, "actions": actions, "state": {
                    "fishPos": p.fish_pos, "fishTargetPos": p.fish_target, "fishSpeed": p.fish_speed,
                    "ypos": p.bar_top, "barSpeed": p.bar_speed, "difficulty": p.difficulty,
                    "motionType": BEHAVIORS.index(p.behavior), "length": p.height,
                    "bobber": upstream_tackle[tackle],
                }})
                rows = []
                for held in actions:
                    p.progress = 0.5  # Keep the raw numerical comparison running.
                    p.step(held)
                    rows.append([p.fish_pos, p.fish_target, p.fish_speed, p.bar_top, p.bar_speed])
                expected.append(rows)
        process = subprocess.run(["node", str(ROOT / "reference_check.mjs")],
                                 input=json.dumps({"source": str(source), "cases": cases}),
                                 capture_output=True, text=True, check=True)
        actual = json.loads(process.stdout)
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-10)
        print("Reference parity: 25 fish/tackle cases x 400 physics ticks passed.", flush=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
