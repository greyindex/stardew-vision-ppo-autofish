"""Read-only inference helpers. No screenshot capture, mouse input or game memory."""
from pathlib import Path
import cv2
import numpy as np
import torch
from . import WIDTH, HEIGHT, SCHEMA, ROWS, PRESENCE
from .model import FishingVision


def letterbox(bgr):
    h, w = bgr.shape[:2]
    scale = min(WIDTH/w, HEIGHT/h)
    tx, ty = (WIDTH-w*scale)/2, (HEIGHT-h*scale)/2
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    border = tuple(map(int, np.median(rgb[:, 0], axis=0)))
    matrix = np.array([[scale, 0, tx], [0, scale, ty]], np.float32)
    result = cv2.warpAffine(rgb, matrix, (WIDTH, HEIGHT), flags=cv2.INTER_LINEAR,
                           borderValue=border)
    return result, (scale, tx, ty)


class VisionPredictor:
    def __init__(self, checkpoint, device="cuda"):
        self.device = torch.device(device)
        checkpoint = torch.load(Path(checkpoint), map_location="cpu", weights_only=True)
        if checkpoint["schema"] != SCHEMA:
            raise ValueError("Unsupported vision schema")
        self.model = FishingVision(checkpoint["channels"]).to(self.device, memory_format=torch.channels_last)
        self.model.load_state_dict(checkpoint["model"])
        self.model.eval()
        self.step = checkpoint["step"]

    @torch.inference_mode()
    def predict(self, crops):
        prepared = [letterbox(crop) for crop in crops]
        array = np.stack([r[0].transpose(2, 0, 1) for r in prepared])
        images = torch.from_numpy(array).to(self.device).float()/255
        images = images.contiguous(memory_format=torch.channels_last)
        with torch.autocast(device_type=self.device.type, dtype=torch.float16,
                            enabled=self.device.type == "cuda"):
            prediction = self.model(images)
        row_prob = prediction["rows"].float().softmax(-1).cpu().numpy()
        visible = prediction["presence"].float().sigmoid().cpu().numpy()
        semantic = prediction["seg"].argmax(1).cpu().numpy().astype(np.uint8)
        results = []
        for i, (_, (scale, tx, ty)) in enumerate(prepared):
            estimates, spread = [], []
            for p in row_prob[i]:
                peak = int(p.argmax()); lower, upper = max(0, peak-2), min(HEIGHT, peak+3)
                local = p[lower:upper]; y = float(np.dot(np.arange(lower, upper), local)/max(float(local.sum()), 1e-9))
                estimates.append(float((y-ty)/scale))
                spread.append(float(1-p[max(0, peak-4):min(HEIGHT, peak+5)].sum()))
            result = {"schema": SCHEMA, "checkpoint_step": self.step,
                      "rows": dict(zip(ROWS, estimates)),
                      "presence_scores": dict(zip(PRESENCE, map(float, visible[i]))),
                      "row_mass_outside_peak": dict(zip(ROWS, spread)),
                      "coordinates": "input_crop_pixels; fish is visible sprite center, not game hitbox",
                      "scores_calibrated": False}
            # A crop classifier's raw sigmoid is not a calibrated confidence.
            result["panel_candidate"] = bool(visible[i, 0] >= .8)
            result["geometry_candidate"] = bool(result["panel_candidate"] and min(visible[i, 1:]) >= .7
                                                  and 0 < estimates[2]-estimates[1] < crops[i].shape[0]*.6)
            result["mask"] = semantic[i]
            result["transform"] = (scale, tx, ty)
            results.append(result)
        return results


class PanelLocator:
    """Low-frequency structural proposal; the neural classifier must confirm it.

    Wood rails at both sides and the lane separate this proposal from a single
    energy/health gauge. A returned matching score is not a probability.
    """
    def __init__(self, menu, max_width=960):
        self.menu = cv2.imread(str(menu))
        if self.menu is None:
            raise FileNotFoundError(menu)
        self.max_width = max_width
        mask = np.zeros(self.menu.shape[:2], np.uint8)
        mask[3:294, 25:32] = 255; mask[3:294, 53:61] = 255
        self.mask = mask

    def locate(self, bgr, scales=(1., 1.5, 2., 2.5, 3., 4.)):
        h, w = bgr.shape[:2]; reduction = min(1., self.max_width/w)
        small = cv2.resize(bgr, None, fx=reduction, fy=reduction, interpolation=cv2.INTER_AREA)
        candidates = []
        for scale in scales:
            size = (round(94*scale*reduction), round(300*scale*reduction))
            if min(size) < 8 or size[0] > small.shape[1] or size[1] > small.shape[0]: continue
            template = cv2.resize(self.menu, size, interpolation=cv2.INTER_AREA)
            mask = cv2.resize(self.mask, size, interpolation=cv2.INTER_NEAREST)
            match = cv2.matchTemplate(small, template, cv2.TM_CCORR_NORMED, mask=mask)
            match = np.nan_to_num(match, nan=-1., posinf=-1., neginf=-1.)
            _, score, _, xy = cv2.minMaxLoc(match)
            candidates.append((score, scale, xy))
        if not candidates: return None
        score, scale, (x, y) = max(candidates)
        if score < .96: return None
        return {"left": int(round(x/reduction)), "top": int(round(y/reduction)),
                "width": int(round(94*scale)), "height": int(round(300*scale)),
                "structural_score": float(score)}
