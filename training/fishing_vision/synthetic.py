"""Pinned original sprites + randomized rendering, with renderer-derived labels.

No labels from the old HSV detector enter this training dataset.
"""
from pathlib import Path
import json
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from . import WIDTH, HEIGHT


def composite(canvas, sprite, x, y, semantic=None, label=0, opacity=1.0):
    x, y = int(round(x)), int(round(y))
    h, w = sprite.shape[:2]
    x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x+w, canvas.shape[1]), min(y+h, canvas.shape[0])
    if x1 <= x0 or y1 <= y0:
        return
    foreground = sprite[y0-y:y1-y, x0-x:x1-x]
    alpha = foreground[..., 3:4].astype(np.float32) / 255 * opacity
    canvas[y0:y1, x0:x1] = (foreground[..., :3]*alpha + canvas[y0:y1, x0:x1]*(1-alpha)).astype(np.uint8)
    if semantic is not None:
        region = semantic[y0:y1, x0:x1]
        region[alpha[..., 0] > 0.08] = label


class SyntheticFishing(Dataset):
    def __init__(self, assets, negatives=None, length=100000, seed=12345):
        self.assets = Path(assets)
        self.length, self.seed = int(length), int(seed)
        self.menu = self._image("fishing_menu.png")
        self.fish = self._image("fish.png")
        legendary = self.assets / "legendary_prototype.png"
        self.legend = self._image(legendary.name) if legendary.exists() else None
        self.treasure = self._image("treasure.png")
        self.negatives = []
        if negatives:
            for path in sorted(Path(negatives).glob("*.png")):
                bgr = cv2.imread(str(path))
                if bgr is not None:
                    self.negatives.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

    def _image(self, name):
        image = cv2.imread(str(self.assets / name), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise FileNotFoundError(self.assets / name)
        return cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)

    def __len__(self):
        return self.length

    def render(self, index):
        rng = np.random.default_rng(self.seed + int(index))
        h, w = 300, 94
        bg = rng.integers(15, 165, (12, 5, 3), dtype=np.uint8)
        canvas = cv2.resize(bg, (w, h), interpolation=cv2.INTER_CUBIC)
        mask = np.zeros((h, w), dtype=np.uint8)
        rows = np.zeros(4, np.float32)
        row_valid = np.zeros(4, np.float32)
        presence = np.zeros(4, np.float32)
        negative = rng.random() < .18
        kind = "negative"
        if not negative:
            composite(canvas, self.menu, 0, 0)
            presence[0] = 1
            level = int(rng.integers(0, 21))
            length = 48 + 4*level + (12 if rng.random() < .15 else 0)
            top = float(rng.uniform(6, 288-length))
            opacity = float(rng.uniform(.08, .75) if rng.random() < .55 else 1.)
            # Match source sprite borders instead of a flat green rectangle.
            layer = np.zeros((300, 94, 4), np.uint8)
            def rect(x, y, ww, hh, rgb):
                yy = int(round(y)); layer[yy:yy+int(hh), x:x+ww] = (*rgb, 255)
            rect(33, top+2, 18, length-4, (33, 101, 1))
            rect(35, top+4, 14, length-8, (130, 229, 0))
            rect(35, top, 14, 2, (73, 193, 0))
            rect(35, top+2, 14, 2, (186, 255, 89))
            rect(35, top+length-4, 14, 2, (73, 193, 0))
            rect(35, top+length-2, 14, 2, (33, 101, 1))
            composite(canvas, layer, 0, 0, mask, 3, opacity)
            rows[1:3] = top, top+length
            presence[2] = row_valid[1] = row_valid[2] = float(opacity >= .16)
            progress = float(rng.uniform(.003, 1))
            yprogress = int(round(292-288*progress))
            rgb = (255, int(progress*510), 0) if progress < .5 else (int((1-progress)*510), 255, 0)
            canvas[yprogress:292, 63:70] = rgb
            mask[yprogress:292, 63:70] = 4
            rows[3] = yprogress
            presence[3] = row_valid[3] = 1
            if rng.random() < .4:
                composite(canvas, self.treasure, 32+rng.integers(-1, 2), rng.integers(10, 252), mask, 5)
            legend = self.legend is not None and rng.random() < .2
            fish = self.legend if legend else self.fish
            # Frequent hard overlaps, and also independent heights for recovery.
            fish_y = float(np.clip(top+length/2-10+rng.uniform(-length, length), 0, 269)) if rng.random() < .55 else float(rng.uniform(0, 269))
            if rng.random() > .025:
                composite(canvas, fish, 32+rng.integers(-1, 2), fish_y, mask, 2 if legend else 1)
                rows[0] = round(fish_y) + 10
                presence[1] = row_valid[0] = 1
            kind = "legendary_prototype" if legend else ("faded" if opacity < 1 else "normal")
        elif self.negatives and rng.random() < .55:
            canvas = cv2.resize(self.negatives[int(rng.integers(len(self.negatives)))], (w, h))
        else:
            # Energy, health and casting gauges have no fish/panel labels.
            for _ in range(int(rng.integers(1, 4))):
                x = int(rng.integers(4, 70)); y = int(rng.integers(0, 70))
                ww, hh = int(rng.integers(8, 19)), int(rng.integers(120, 230))
                canvas[y:min(y+hh, h), x:x+ww] = (126, 75, 30)
                fill = int(hh*rng.uniform(.05, .99))
                canvas[max(y, y+hh-fill):min(y+hh-2, h), x+2:x+ww-2] = (40, 230, 50) if rng.random() < .5 else (235, 45, 20)
        # Independent scale and placement jitter. Track inverse transform labels.
        scale = min(WIDTH/w, HEIGHT/h)*float(rng.uniform(.87, 1.01))
        ww, hh = max(1, round(w*scale)), max(1, round(h*scale))
        tx = int((WIDTH-ww)/2 + rng.integers(-5, 6))
        ty = int((HEIGHT-hh)/2 + rng.integers(-8, 9))
        matrix = np.array([[scale, 0, tx], [0, scale, ty]], np.float32)
        image = cv2.warpAffine(canvas, matrix, (WIDTH, HEIGHT), flags=cv2.INTER_NEAREST,
                               borderMode=cv2.BORDER_CONSTANT,
                               borderValue=tuple(map(int, canvas[h//2, 0])))
        label = cv2.warpAffine(mask, matrix, (WIDTH, HEIGHT), flags=cv2.INTER_NEAREST)
        rows = rows*scale+ty
        row_valid *= ((rows >= 0) & (rows < HEIGHT)).astype(np.float32)
        gain = rng.uniform(.82, 1.15, (1, 1, 3))
        image = np.clip(image.astype(np.float32)*gain + rng.normal(0, rng.uniform(0, 2.5), image.shape), 0, 255).astype(np.uint8)
        if rng.random() < .25:
            image = cv2.GaussianBlur(image, (3, 3), .45)
        if rng.random() < .3:
            ok, encoded = cv2.imencode(".jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(60, 96))])
            if ok:
                image = cv2.cvtColor(cv2.imdecode(encoded, 1), cv2.COLOR_BGR2RGB)
        return image, label, rows, row_valid, presence, kind

    def __getitem__(self, index):
        image, mask, rows, valid, presence, _ = self.render(index)
        return {"image": torch.from_numpy(image.transpose(2, 0, 1).copy()).float()/255,
                "mask": torch.from_numpy(mask.astype(np.int64)),
                "rows": torch.from_numpy(rows), "row_valid": torch.from_numpy(valid),
                "presence": torch.from_numpy(presence)}


class LabelledCrops(Dataset):
    """Manually reviewed geometry; unlabelled pixels do not become background GT."""
    def __init__(self, manifest):
        self.manifest = Path(manifest)
        data = json.loads(self.manifest.read_text(encoding="utf-8"))
        self.items = [r for r in data["frames"] if r.get("reviewed") and r.get("split") == "train"]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        item = self.items[index]
        bgr = cv2.imread(str(self.manifest.parent / item["image"]))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]; scale = min(WIDTH/w, HEIGHT/h)
        tx, ty = (WIDTH-w*scale)/2, (HEIGHT-h*scale)/2
        matrix = np.array([[scale, 0, tx], [0, scale, ty]], np.float32)
        rgb = cv2.warpAffine(rgb, matrix, (WIDTH, HEIGHT), flags=cv2.INTER_LINEAR)
        return {"image": torch.from_numpy(rgb.transpose(2, 0, 1).copy()).float()/255,
                "mask": torch.full((HEIGHT, WIDTH), -1, dtype=torch.int64),
                "rows": torch.tensor(item["rows"], dtype=torch.float32)*scale+ty,
                "row_valid": torch.tensor(item["row_valid"], dtype=torch.float32),
                "presence": torch.tensor(item["presence"], dtype=torch.float32)}
