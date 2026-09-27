"""Screen-space bite marker detection without a player or weather template."""
import cv2
import numpy as np


class BiteMarkerReader:
    """Find a new gold stem/dot pair outside the fixed upper-right HUD.

    This supplies spatial evidence; the cycle must still confirm persistence
    across fresh frames and only enable it while waiting for a bite.
    """
    def __init__(self, baseline):
        self.size = (min(1920, baseline.shape[1]),
                     round(baseline.shape[0]*min(1., 1920/baseline.shape[1])))
        self.scale = baseline.shape[1]/self.size[0]
        self.baseline = cv2.resize(baseline, self.size, interpolation=cv2.INTER_AREA)

    def set_baseline(self, image):
        self.baseline = cv2.resize(image, self.size, interpolation=cv2.INTER_AREA)

    @staticmethod
    def bright_core(image, mask):
        value = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)[:, :, 2]
        _, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
        core = mask.copy()
        for label, (x, y, w, h, area) in enumerate(stats[1:], 1):
            if not (1.7 <= h/max(w, 1) <= 8 and 8 <= h <= 100 and area >= 8):
                continue
            selected = labels[y:y+h, x:x+w] == label
            patch = value[y:y+h, x:x+w]
            levels = patch[selected]
            low, high = np.percentile(levels, (20, 90))
            if high-low >= 25:
                threshold, _ = cv2.threshold(levels, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
                core[y:y+h, x:x+w][selected & (patch <= threshold)] = 0
        return core

    @staticmethod
    def pairs(mask):
        _, _, stats, _ = cv2.connectedComponentsWithStats(mask)
        pieces = stats[1:]
        stems = pieces[(pieces[:, 3] >= 6) & (pieces[:, 3] <= 90)
                       & (pieces[:, 3] >= 2*pieces[:, 2]) & (pieces[:, 3] <= 8*pieces[:, 2])
                       & (pieces[:, 4] >= .55*pieces[:, 2]*pieces[:, 3])]
        for x, y, w, h, area in stems:
            dots = pieces[(pieces[:, 1] >= y+h) & (pieces[:, 1] <= y+h+max(3, 2*w))
                          & (np.abs(pieces[:, 0]+pieces[:, 2]/2-(x+w/2)) <= max(1.5, .75*w))
                          & (pieces[:, 2] >= .45*w) & (pieces[:, 2] <= 2*w)
                          & (pieces[:, 3] >= .45*w) & (pieces[:, 3] <= 2*w)
                          & (pieces[:, 4] >= 2)
                          & (pieces[:, 4] >= .40*pieces[:, 2]*pieces[:, 3])]
            for dot in dots:
                yield tuple(map(int, (x, y, w, h))), tuple(map(int, dot[:4]))

    @staticmethod
    def outlined_stem(frame, stem):
        """The gold glyph has a dark warm outline on both vertical sides."""
        x, y, w, h = stem
        margin = max(2, w)
        if x < margin or x+w+margin > frame.shape[1]:
            return False
        inset = max(1, round(h*.15))
        core = float(np.median(frame[y+inset:y+h-inset, x:x+w].max(2)))
        for left, right in ((x-margin, x), (x+w, x+w+margin)):
            rim = frame[y+inset:y+h-inset, left:right].astype(np.float32)
            blue, green, red = cv2.split(rim)
            brown = ((red >= green*1.05) & (green >= blue*.7)
                     & (rim.max(2) < core*.85))
            if float(brown.mean()) < .55:
                return False
        return True

    def detect(self, frame):
        small = cv2.resize(frame, self.size, interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (18, 55, 55), (40, 255, 255))
        novel = cv2.inRange(cv2.absdiff(small, self.baseline), (0, 0, 0), (17, 17, 17)) == 0
        mask[~novel] = 0
        # Quest !, clock and journal buttons are fixed HUD, never bite evidence.
        xhud, yhud = round(self.size[0]*.84), round(self.size[1]*.28)
        mask[:yhud, xhud:] = 0
        evidence = {"visible": False, "scope": "screen", "baseline": "screen_wait",
                    "excluded_region": [round(xhud*self.scale), 0,
                                        frame.shape[1]-round(xhud*self.scale), round(yhud*self.scale)]}
        for mode in ("gold", "bright_core"):
            selected = mask if mode == "gold" else self.bright_core(small, mask)
            for stem, dot in self.pairs(selected):
                x, y, w, h = stem
                xx, yy, ww, hh = dot
                # Brown trousers/rod segments and lime leaves can form vertical
                # pairs too. Compare hue, not RGB intensity: the real dot can
                # be substantially dimmer than the stem during its fade.
                hues = [np.median(hsv[py:py+ph, px:px+pw, 0][selected[py:py+ph, px:px+pw] > 0])
                        for px, py, pw, ph in (stem, dot)]
                if not all(20 <= hue <= 38 for hue in hues) or abs(hues[0]-hues[1]) > 14:
                    continue
                convert = lambda box: [round(v*self.scale) for v in box]
                if not self.outlined_stem(frame, convert(stem)):
                    continue
                evidence.update(visible=True, stem=convert(stem), dot=convert(dot), mask_mode=mode,
                                region=convert((min(x, xx), y, max(x+w, xx+ww)-min(x, xx), yy+hh-y)))
                return evidence
        return evidence
