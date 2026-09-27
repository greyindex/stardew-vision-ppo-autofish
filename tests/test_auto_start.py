"""Startup regressions; all captures and mouse calls are in-memory fakes."""
import copy
from pathlib import Path
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import cv2
import numpy as np

from auto_fishing import AutoFishingCycle, GaugeReader

ROOT = Path(__file__).resolve().parents[1]


def prediction(complete=True):
    mask = np.zeros((300, 94), np.uint8)
    if complete:
        mask[130:178, 33:51] = 3
        mask[200:292, 63:70] = 4
    mask[143:158, 35:49] = 1
    return {"transform": (1., 0., 0.), "mask": mask,
            "rows": {"fish_visual_center": 150., "bar_top": 130., "bar_bottom": 178., "progress_top": 200.},
            "presence_scores": {"panel": .999, "fish": .999, "bar": .99, "progress": .99}}


def cycle_for(pred=None, energy=(None,)):
    frame = np.zeros((300, 94, 3), np.uint8)
    client = {"left": 0, "top": 0, "width": 94, "height": 300}
    active = threading.Event(); active.set()
    mouse = SimpleNamespace(holding=False, move=Mock())

    def hold(seconds, hwnd):
        mouse.holding = True
        return time.perf_counter()

    mouse.hold_for = Mock(side_effect=hold)
    stop = Mock(side_effect=lambda message: active.clear())
    matcher = SimpleNamespace(locate=lambda image: dict(client) if pred is not None else None,
                              score=lambda image: 1., inference_crop=lambda image: image,
                              fish_sprite_center=lambda image: (None, 0.))
    session = SimpleNamespace(generation=1, active=active, mouse=mouse, matcher=matcher,
                              gauge_reader=SimpleNamespace(energy_fraction=Mock(side_effect=energy)),
                              vision=SimpleNamespace(predict=lambda images: [copy.deepcopy(pred)]),
                              _capture=lambda sct, rect: (frame.copy(), time.perf_counter()),
                              _record=Mock(), note=Mock(), publish=Mock(), stop=stop)
    config = {"auto_bite": "visual", "fish_offset_native": 1.,
              "auto_origin": {"x": 30, "y": 200, "client_size": [94, 300]}}
    cycle = AutoFishingCycle(session, config)
    cycle.entered = time.perf_counter() - .5
    return cycle, session, client


class StartGuardChecks(unittest.TestCase):
    def test_panel_and_fish_lookalike_does_not_block_cast(self):
        cycle, session, client = cycle_for(prediction(complete=False))
        cycle.tick(None, client, 1)
        session.mouse.hold_for.assert_called_once_with(1.05, 1)
        session.stop.assert_not_called()
        self.assertEqual(cycle.phase, "charge")

    def test_complete_minigame_still_blocks_cast(self):
        cycle, session, client = cycle_for(prediction())
        cycle.tick(None, client, 1)
        session.mouse.hold_for.assert_not_called()
        session.stop.assert_called_once()

    def test_minigame_with_occluded_fish_still_blocks_cast(self):
        pred = prediction()
        pred["mask"][pred["mask"] == 1] = 5
        pred["presence_scores"]["fish"] = .1
        cycle, session, client = cycle_for(pred)
        cycle.tick(None, client, 1)
        session.mouse.hold_for.assert_not_called()
        session.stop.assert_called_once()

    def test_single_low_reading_cannot_block_full_energy(self):
        cycle, session, client = cycle_for(energy=(.05, .85))
        cycle.tick(None, client, 1)
        session.mouse.hold_for.assert_called_once()
        session.stop.assert_not_called()

    def test_unreadable_confirmation_is_not_zero(self):
        cycle, session, client = cycle_for(energy=(.05, None))
        cycle.tick(None, client, 1)
        session.mouse.hold_for.assert_called_once()
        session.stop.assert_not_called()

    def test_confirmed_low_energy_still_blocks_cast(self):
        cycle, session, client = cycle_for(energy=(.05, .06))
        cycle.tick(None, client, 1)
        session.mouse.hold_for.assert_not_called()
        session.stop.assert_called_once()


class EnergyReadingChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reader = GaugeReader(ROOT / "auto_assets")

    def test_bright_dark_and_desaturated_full_meter(self):
        full = self.reader.energy
        hsv = cv2.cvtColor(full, cv2.COLOR_BGR2HSV)
        hsv[:, :, 1] = (hsv[:, :, 1].astype(float) * .60).astype(np.uint8)
        daylight = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        night = (full.astype(float) * .5).astype(np.uint8)
        for image in (full, daylight, night):
            with self.subTest(mean=float(image.mean())):
                self.assertGreater(self.reader.energy_crop_fraction(image), .95)

    def test_unreadable_fill_is_unknown(self):
        image = self.reader.energy.copy()
        image[52:216, 11:33] = 80
        self.assertIsNone(self.reader.energy_crop_fraction(image))

    def test_unrelated_label_is_rejected(self):
        image = self.reader.energy.copy()
        image[7:42, 10:36] = (20, 80, 120)
        self.assertIsNone(self.reader.energy_crop_fraction(image))

    def test_low_red_fill_remains_readable(self):
        image = self.reader.energy.copy()
        image[52:216, 11:33] = 30
        image[206:216, 11:33] = (0, 0, 255)
        value = self.reader.energy_crop_fraction(image)
        self.assertGreater(value, 0.)
        self.assertLess(value, .12)


if __name__ == "__main__":
    unittest.main()
