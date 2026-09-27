"""Local simulator and training dashboard; never sends keyboard/mouse events to a game."""
import argparse
import ctypes
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import traceback
from urllib.parse import urlparse
import torch
from stable_baselines3 import PPO
from .core import CATALOG, TACKLES, LANE_TOP, LANE_HEIGHT
from .env import FishingEnv
from .controllers import MPC, PredictiveRule

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
ACTIVE_STATES = {"training", "evaluating", "stopping"}


def read_json(path, fallback=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return fallback


def alive(pid):
    if not isinstance(pid, int) or pid <= 0:
        return False
    if os.name == "nt":
        kernel = ctypes.windll.kernel32
        kernel.OpenProcess.restype = ctypes.c_void_p
        kernel.GetExitCodeProcess.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong)]
        kernel.CloseHandle.argtypes = [ctypes.c_void_p]
        handle = kernel.OpenProcess(0x1000, False, pid)
        if not handle:
            return False
        code = ctypes.c_ulong()
        ok = kernel.GetExitCodeProcess(handle, ctypes.byref(code))
        kernel.CloseHandle(handle)
        return bool(ok and code.value == 259)
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def all_runs():
    return sorted((p.parent for p in RUNS.glob("*/manifest.json")), key=lambda p: (p / "manifest.json").stat().st_mtime, reverse=True)


def latest_run():
    runs = all_runs()
    return runs[0] if runs else None


def tail(path, limit=200_000):
    try:
        with Path(path).open("rb") as f:
            size = f.seek(0, 2)
            f.seek(max(0, size-limit))
            data = f.read().decode("utf-8", errors="replace")
        return data.split("\n", 1)[-1] if size > limit else data
    except OSError:
        return ""


class Dashboard:
    def __init__(self):
        self.lock = threading.RLock()
        self.training_process = None
        self.model = None
        self.model_signature = None
        self.mpc, self.rule = MPC(), PredictiveRule()
        self.settings = {"fish": "Largemouth Bass", "level": 5, "tackle": "none", "seed": 123, "controller": "mpc"}
        self.reset(self.settings)

    def load_model(self):
        run = latest_run()
        choices = [run / name for name in ("best.zip", "latest.zip", "initial.zip")] if run else []
        checkpoint = next((p for p in choices if p.exists()), None)
        if checkpoint is None:
            raise ValueError("还没有模型文件，请先开始训练。")
        signature = (str(checkpoint), checkpoint.stat().st_mtime_ns)
        if signature != self.model_signature:
            self.model = PPO.load(checkpoint, device="cpu")
            self.model_signature = signature

    def reset(self, data):
        settings = {**self.settings, **{k:v for k,v in data.items() if k in self.settings}}
        if settings["controller"] not in ("mpc", "rule", "ppo", "manual"):
            raise ValueError("Unsupported controller")
        settings["level"] = int(settings["level"])
        settings["seed"] = int(settings["seed"])
        if not 0 <= settings["seed"] < 2**31:
            raise ValueError("seed must be between 0 and 2147483647")
        env = FishingEnv(fish=settings["fish"], level=settings["level"], tackle=settings["tackle"])
        observation, _ = env.reset(seed=settings["seed"])
        if settings["controller"] == "ppo":
            self.load_model()
        self.settings, self.env, self.observation = settings, env, observation
        self.held = 0
        return self.snapshot()

    def tick(self, data):
        steps = max(1, min(int(data.get("steps", 1)), 30))
        for _ in range(steps):
            if self.env.physics.done:
                break
            mode = self.settings["controller"]
            if mode == "ppo":
                action = int(self.model.predict(self.observation, deterministic=True)[0])
            elif mode == "manual":
                action = int(bool(data.get("held", False)))
            else:
                action = (self.mpc if mode == "mpc" else self.rule)(self.observation)
            self.held = action
            self.observation, _, _, _, _ = self.env.step(action)
        return self.snapshot()

    def snapshot(self):
        p = self.env.physics
        return {**p.measurement(), **p.diagnostics(), "settings": self.settings,
                "held": self.held, "done": p.done, "contained": p.contained,
                "lane_top": LANE_TOP, "lane_height": LANE_HEIGHT,
                "model": str(self.model_signature[0]) if self.model_signature else None}

    def status(self):
        run = latest_run()
        state = read_json(run / "status.json", {}) if run else {}
        if state.get("state") in ACTIVE_STATES and not alive(state.get("pid")):
            state = {**state, "state": "interrupted", "error": "训练进程已退出。可从检查点继续训练。"}
        curve = {}
        if run:
            # Per-evaluation files retain the whole curve when long training logs are tailed.
            for path in sorted(run.glob("eval_*.json")):
                ev = read_json(path)
                if ev and "steps" in ev and "catch_rate" in ev:
                    curve[ev["steps"]] = {"steps": ev["steps"], "catch_rate": ev["catch_rate"]}
        baselines = {}
        manifest = read_json(run / "manifest.json", {}) if run else {}
        evaluation_seed = manifest.get("development_seed", 1_000_000)
        evaluation_count = manifest.get("arguments", {}).get("eval_episodes", 80)
        for name in ("rule", "mpc"):
            result = read_json(RUNS / f"baseline_{name}.json")
            seeds = sorted(row["seed"] for row in result.get("episodes_detail", [])) if result else []
            expected = list(range(evaluation_seed, evaluation_seed + evaluation_count))
            if result and seeds == expected:
                baselines[name] = {k:v for k,v in result.items() if k != "episodes_detail"}
        final = read_json(run / "final_evaluation.json") if run else None
        if final:
            final = {k:v for k,v in final.items() if k != "episodes_detail"}
        return {"training": state, "curve": list(curve.values()), "baselines": baselines, "final_evaluation": final,
                "run_name": run.name if run else None,
                "log": tail(run / "stdout.log", 5000) if run else "",
                "error_log": tail(run / "stderr.log", 3000) if run else ""}

    def start_training(self, data):
        for run in all_runs():
            status = read_json(run / "status.json", {})
            if status.get("state") in ACTIVE_STATES and alive(status.get("pid")):
                raise ValueError("已有训练正在运行。")
        if self.training_process and self.training_process.poll() is None:
            raise ValueError("训练进程正在启动。")
        steps = int(data.get("steps", 2_000_000))
        if not 10_000 <= steps <= 10_000_000:
            raise ValueError("训练步数范围为 10,000 到 10,000,000。")
        run = RUNS / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        run.mkdir(parents=True)
        args = [sys.executable, "-u", "-m", "fishing_sim.train", "--steps", str(steps), "--run-dir", str(run)]
        if data.get("resume"):
            previous = latest_run()
            candidates = [previous / name for name in ("latest.zip", "final.zip", "best.zip")] if previous else []
            checkpoint = next((p for p in candidates if p.exists()), None)
            if checkpoint is None:
                raise ValueError("找不到可继续训练的检查点。")
            args += ["--resume", str(checkpoint), "--no-curriculum"]
        with (run / "stdout.log").open("w", encoding="utf-8") as out, (run / "stderr.log").open("w", encoding="utf-8") as err:
            self.training_process = subprocess.Popen(args, cwd=ROOT, stdout=out, stderr=err,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        return {"pid": self.training_process.pid, "run": str(run)}

    def stop_training(self):
        for run in all_runs():
            status = read_json(run / "status.json", {})
            if status.get("state") in ACTIVE_STATES and alive(status.get("pid")):
                (run / "STOP").write_text("Requested from dashboard\n", encoding="utf-8")
                return {"state": "stopping"}
        return {"state": "idle"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8767)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    app = Dashboard()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):
            pass

        def respond(self, data, code=200, content_type="application/json; charset=utf-8"):
            payload = data if isinstance(data, bytes) else json.dumps(data, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            try:
                self.wfile.write(payload)
            except (BrokenPipeError, ConnectionResetError):
                pass  # A closed/reloaded dashboard is not a simulator failure.

        def do_GET(self):
            path = urlparse(self.path).path
            if path == "/":
                self.respond((ROOT / "dashboard.html").read_bytes(), content_type="text/html; charset=utf-8")
            elif path == "/api/status":
                self.respond(app.status())
            elif path == "/api/catalog":
                self.respond({"fish": CATALOG, "tackles": TACKLES})
            elif path == "/api/state":
                with app.lock:
                    self.respond(app.snapshot())
            else:
                self.respond({"error": "Not found"}, 404)

        def do_POST(self):
            origin = self.headers.get("Origin")
            if origin and origin not in (f"http://127.0.0.1:{args.port}", f"http://localhost:{args.port}"):
                self.respond({"error": "Invalid origin"}, 403)
                return
            if not self.headers.get("Content-Type", "").startswith("application/json"):
                self.respond({"error": "JSON required"}, 415)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 <= length <= 10000:
                    raise ValueError("Request too large")
                data = json.loads(self.rfile.read(length) or b"{}")
                with app.lock:
                    method = {"/api/reset": app.reset, "/api/tick": app.tick, "/api/train": app.start_training}.get(self.path)
                    if method:
                        result = method(data)
                    elif self.path == "/api/stop-training":
                        result = app.stop_training()
                    else:
                        self.respond({"error": "Not found"}, 404)
                        return
                self.respond(result)
            except (ValueError, KeyError, TypeError) as exc:
                self.respond({"error": str(exc)}, 400)
            except Exception as exc:
                traceback.print_exc()
                self.respond({"error": str(exc)}, 500)

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    (ROOT / ".server.json").write_text(json.dumps({"pid": os.getpid(), "port": args.port}), encoding="utf-8")
    print(f"Fishing simulator: http://127.0.0.1:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
