"""Visual measurements for fishing. Training and replay never send game input."""

SCHEMA = "fishing-vision-v1"
WIDTH, HEIGHT = 192, 640
CLASSES = ("background", "fish", "legendary_fish", "bar", "progress", "treasure")
ROWS = ("fish_visual_center", "bar_top", "bar_bottom", "progress_top")
PRESENCE = ("panel", "fish", "bar", "progress")
