"""Color-independent catch-card evidence for tinted Stardew Valley scenes."""
import cv2
import numpy as np


def tinted_card(image, baseline, novel, center, meter_width, meter_height):
    """Find a newly drawn, flat card surface with a new horizontal lower edge.

    At night the white card can become saturated blue. Broadening a white HSV
    threshold would also admit water and foliage, so require a coherent fill,
    card geometry, and an edge absent from the pre-cast scene instead.
    The top/left of the card may be clipped by the player's observation ROI.
    """
    minimum = .45*meter_width*meter_height
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    changed = novel & (hsv[:, :, 2] >= 85)
    if int(changed.sum()) < minimum:
        return {"visible": False}
    quantized = (image//32).astype(np.int32)
    keys = quantized[:, :, 0]*64+quantized[:, :, 1]*8+quantized[:, :, 2]
    counts = np.bincount(keys[changed], minlength=512)
    colors = []
    kernel = max(1, round(meter_height/24))
    band = max(1, round(meter_height/24))
    current = image.astype(np.float32)
    before = baseline.astype(np.float32)
    best_edge = 0.
    candidates = 0
    for key in counts.argsort()[-10:][::-1]:
        if counts[key] < minimum*.20:
            break
        color = np.median(image[changed & (keys == key)], axis=0)
        if any(np.abs(color-other).max() <= 22 for other in colors):
            continue
        colors.append(color)
        selected = ((np.abs(current-color).max(2) <= 22) & novel).astype(np.uint8)*255
        selected = cv2.morphologyEx(selected, cv2.MORPH_CLOSE, np.ones((kernel, kernel), np.uint8))
        _, labels, stats, _ = cv2.connectedComponentsWithStats(selected)
        for label, (x, y, w, h, area) in enumerate(stats[1:], 1):
            if not (w >= .65*meter_width and h >= .55*meter_height and x <= center <= x+w
                    and area >= minimum and area >= .55*w*h):
                continue
            candidates += 1
            component = labels[y:y+h, x:x+w] == label
            # Text/icon holes are allowed. Most of the bottom of a card is one
            # straight surface; the narrow speech pointer is below that line.
            broad_rows = np.flatnonzero(component.mean(1) >= .65)
            if len(broad_rows) < .45*h:
                continue
            bottom = int(y)+int(broad_rows[-1])+1
            if bottom < band or bottom+3*band > image.shape[0]:
                continue
            pad = max(1, round(w*.04))
            left, right = x+pad, x+w-pad
            upper = current[bottom-band:bottom, left:right].mean(0)
            lower = current[bottom+band:bottom+3*band, left:right].mean(0)
            prior_upper = before[bottom-band:bottom, left:right].mean(0)
            prior_lower = before[bottom+band:bottom+3*band, left:right].mean(0)
            contrast = (upper-lower).max(1)
            prior_contrast = (prior_upper-prior_lower).max(1)
            edge = float(((contrast >= 20) & (contrast >= prior_contrast+12)).mean())
            best_edge = max(best_edge, edge)
            if edge < .60:
                continue
            return {"visible": True, "box": list(map(int, (x, y, w, h))), "paper_area": int(area),
                    "source": "tinted_flat_card", "fill_bgr": color.tolist(),
                    "fill_fraction": float(area/(w*h)), "new_edge_fraction": edge,
                    "bottom_row": bottom}
    return {"visible": False, "flat_candidates": candidates, "best_new_edge_fraction": best_edge}
