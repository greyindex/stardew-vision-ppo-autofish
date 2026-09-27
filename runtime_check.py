"""Read-only runtime diagnostics for the one-click Windows launcher."""
import argparse
import importlib.metadata
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
TRAINING = ROOT / "training"
sys.path.insert(0, str(TRAINING))


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--quick", action="store_true")
    group.add_argument("--full", action="store_true")
    args = parser.parse_args()
    lock = ROOT / "requirements-vision-gui-lock.txt"
    for line in lock.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        name, expected = line.split("==", 1)
        try:
            installed = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            installed = None
        if installed != expected:
            print(f"Dependency mismatch: {name} expected {expected}, found {installed}", file=sys.stderr)
            return 2
    try:
        import cv2
        import mss  # noqa: F401
        import pydirectinput  # noqa: F401
        import pynput  # noqa: F401
        import soundcard  # noqa: F401
        import stable_baselines3  # noqa: F401
        import torch
        from PIL import Image  # noqa: F401
    except ImportError as exc:
        print(f"Missing or broken dependency: {exc}", file=sys.stderr)
        return 2

    required = [
        ROOT / "training/vision_runs/20260927_v1_synthetic/best.pt",
        ROOT / "training/runs/20260927_scaling_10m/best.zip",
        ROOT / "training/runs/20260927_scaling_10m/manifest.json",
        ROOT / "training/vision_data/assets/fishing_menu.png",
        ROOT / "auto_assets/power_meter.png",
        ROOT / "auto_assets/energy_meter.png",
    ]
    missing = [str(path.relative_to(ROOT)) for path in required if not path.is_file()]
    if missing:
        print("Missing runtime files: " + ", ".join(missing), file=sys.stderr)
        return 3
    if not torch.cuda.is_available():
        print("CUDA is unavailable. This build requires an NVIDIA GPU, current driver, and CUDA PyTorch wheel.", file=sys.stderr)
        return 4
    if args.full:
        from auto_fishing import GaugeReader
        from fishing_vision.inference import VisionPredictor
        from vision_live import FrozenPolicy, PanelMatcher

        assets = ROOT / "training/vision_data/assets"
        model = VisionPredictor(required[0], device="cuda")
        policy = FrozenPolicy(required[1])
        matcher = PanelMatcher(assets)
        GaugeReader(ROOT / "auto_assets")
        image = matcher.inference_crop(matcher.menu)
        prediction = model.predict([image])[0]
        assert len(prediction["rows"]) == 4
        print(f"Vision step {model.step}; PPO step {policy.model.num_timesteps}; "
              f"GPU {torch.cuda.get_device_name()}; OpenCV {cv2.__version__}")
    else:
        print(f"Runtime ready: torch {torch.__version__}, CUDA {torch.cuda.get_device_name()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
