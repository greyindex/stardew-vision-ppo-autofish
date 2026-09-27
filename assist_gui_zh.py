"""简体中文助手：屏幕识别、冻结 PPO / 预测控制器和模拟鼠标输入。"""

import os
import re
import subprocess
import sys
import threading
import traceback
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk
from pynput import keyboard

from assist_gui import AssistGUI
from main import load_config, save_config
from ppo_live import DEFAULT_MODEL, valid_roi


PROJECT_DIR = Path(__file__).resolve().parent
CONTROLLERS = {"PPO · 1000 万步": "ppo", "原预测控制器": "rule"}
TACKLES = {"无渔具（已训练）": "none", "软木塞浮标（未训练）": "cork", "铅制浮标（未训练）": "lead",
           "陷阱浮标（未训练）": "trap", "倒刺钩（未训练）": "barbed"}


class ChineseAssistGUI(AssistGUI):
    def __init__(self):
        self.calibration_process = None
        self.closing = False
        self.hotkey_stopped = False
        super().__init__()
        self.key_listener = keyboard.Listener(on_press=self._on_key)
        self.key_listener.start()

    def _build_ui(self):
        self.root = tk.Tk()
        self.root.title("星露谷钓鱼助手 · PPO")
        self.root.resizable(False, False)
        self.root.geometry("+28+36")

        self.btn = tk.Button(
            self.root,
            text="开始辅助",
            command=self.toggle,
            font=("Microsoft YaHei UI", 16, "bold"),
            bg="#2e9d64",
            fg="white",
            activebackground="#267c50",
            activeforeground="white",
            relief="flat",
            height=2,
        )
        self.btn.pack(fill="x", padx=14, pady=(14, 8))

        self.lbl = tk.Label(
            self.root,
            text="已停止",
            font=("Microsoft YaHei UI", 10),
            fg="#444444",
            wraplength=390,
            justify="center",
        )
        self.lbl.pack(pady=(0, 7))

        settings = tk.Frame(self.root)
        settings.pack(fill="x", padx=14, pady=(0, 7))
        tk.Label(settings, text="控制方式", anchor="w").grid(row=0, column=0, sticky="w", pady=3)
        self.controller_var = tk.StringVar(value=next((label for label, value in CONTROLLERS.items()
                                                       if value == self.cfg.get("controller", "ppo")), "PPO · 1000 万步"))
        self.controller_combo = ttk.Combobox(settings, textvariable=self.controller_var, values=list(CONTROLLERS), state="readonly", width=29)
        self.controller_combo.grid(row=0, column=1, sticky="ew", padx=(8, 0))
        self.controller_combo.bind("<<ComboboxSelected>>", self._save_settings)
        tk.Label(settings, text="钓鱼等级", anchor="w").grid(row=1, column=0, sticky="w", pady=3)
        level = self.cfg.get("ppo_level", "auto")
        self.level_var = tk.StringVar(value="按绿条长度估计" if level == "auto" else str(level))
        self.level_combo = ttk.Combobox(settings, textvariable=self.level_var,
                                        values=["按绿条长度估计", *map(str, range(21))], state="readonly", width=29)
        self.level_combo.grid(row=1, column=1, sticky="ew", padx=(8, 0))
        self.level_combo.bind("<<ComboboxSelected>>", self._save_settings)
        tk.Label(settings, text="渔具", anchor="w").grid(row=2, column=0, sticky="w", pady=3)
        self.tackle_var = tk.StringVar(value=next((label for label, value in TACKLES.items()
                                                  if value == self.cfg.get("ppo_tackle", "none")), "无渔具（已训练）"))
        self.tackle_combo = ttk.Combobox(settings, textvariable=self.tackle_var, values=list(TACKLES), state="readonly", width=29)
        self.tackle_combo.grid(row=2, column=1, sticky="ew", padx=(8, 0))
        self.tackle_combo.bind("<<ComboboxSelected>>", self._save_settings)
        settings.columnconfigure(1, weight=1)

        self.progress_label = tk.Label(self.root, font=("Microsoft YaHei UI", 9), wraplength=410, justify="left")
        self.progress_label.pack(fill="x", padx=14, pady=(0, 6))
        self._show_progress_region()

        self.auto_hook_var = tk.BooleanVar(value=bool(self.cfg.get("auto_hook", True)))
        self.auto_hook_check = tk.Checkbutton(
            self.root,
            text="自动上钩（竹鱼竿等；请开启游戏音效）",
            variable=self.auto_hook_var,
            command=self._save_auto_hook,
            font=("Microsoft YaHei UI", 9),
            anchor="w",
        )
        self.auto_hook_check.pack(fill="x", padx=14, pady=(0, 5))

        actions = tk.Frame(self.root)
        actions.pack(fill="x", padx=14)
        self.cal_btn = tk.Button(
            actions,
            text="选择钓鱼区域",
            command=self.calibrate,
            font=("Microsoft YaHei UI", 9),
        )
        self.cal_btn.pack(side="left", fill="x", expand=True, padx=(0, 4))
        self.progress_btn = tk.Button(actions, text="选择进度槽", command=self.calibrate_progress,
                                      font=("Microsoft YaHei UI", 9))
        self.progress_btn.pack(side="left", fill="x", expand=True, padx=4)
        self.log_btn = tk.Button(
            actions,
            text="打开日志",
            command=self.open_logs,
            font=("Microsoft YaHei UI", 9),
        )
        self.log_btn.pack(side="left", fill="x", expand=True, padx=(4, 0))

        tk.Label(
            self.root,
            text="PPO：30 Hz 决策 · F8 紧急停止\n先开始助手，再切回游戏手动抛竿。\n首测建议等级 0–10、无渔具；实际游戏表现待测。",
            font=("Microsoft YaHei UI", 8),
            fg="#666666",
        ).pack(pady=(8, 12))

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self._poll()

    def _show_progress_region(self):
        roi = self.cfg.get("progress_roi")
        configured = valid_roi(roi)
        self.progress_label.config(text=(f"进度槽已选择：{int(roi['width'])} × {int(roi['height'])} 像素" if configured
                                         else "PPO 首次使用：请先选择右侧完整进度槽（包含空白部分）。"),
                                   fg="#39734c" if configured else "#986300")

    def _save_settings(self, _event=None):
        cfg = load_config()
        cfg.update(controller=CONTROLLERS[self.controller_var.get()],
                   ppo_level="auto" if self.level_var.get() == "按绿条长度估计" else int(self.level_var.get()),
                   ppo_tackle=TACKLES[self.tackle_var.get()])
        cfg.setdefault("ppo_model", DEFAULT_MODEL)
        save_config(cfg)
        self.cfg = cfg

    def _set_controls(self, enabled):
        state = tk.NORMAL if enabled else tk.DISABLED
        for widget in (self.cal_btn, self.progress_btn, self.auto_hook_check):
            widget.config(state=state)
        for widget in (self.controller_combo, self.level_combo, self.tackle_combo):
            widget.config(state="readonly" if enabled else "disabled")

    def _on_key(self, key):
        if key == keyboard.Key.f8 and self.running:
            # Listener thread only changes flags. Tk updates stay on the UI thread.
            self.hotkey_stopped = True
            self.running = False

    def run(self):
        try:
            super().run()
        finally:
            self.key_listener.stop()

    def start(self):
        if self.running:
            return
        if self.thread is not None and self.thread.is_alive():
            self.lbl.config(text="正在释放鼠标，请稍候…", fg="#8a5b00")
            return
        if self.calibration_process is not None and self.calibration_process.poll() is None:
            self.lbl.config(text="请先完成或取消区域选择。", fg="#986300")
            return
        self._save_settings()
        self.cfg = load_config()
        self._show_progress_region()
        if self.cfg.get("controller") == "ppo":
            if not valid_roi(self.cfg.get("roi")):
                messagebox.showinfo("需要钓鱼轨道选区", "请先选择钓鱼轨道的内部可移动区域。", parent=self.root)
                return
            if not valid_roi(self.cfg.get("progress_roi")):
                messagebox.showinfo("需要进度槽选区", "PPO 需要读取捕获进度。请先点击“选择进度槽”，框选右侧从底到顶的完整细长槽，包含未填充部分。", parent=self.root)
                return
            model = Path(self.cfg.get("ppo_model", DEFAULT_MODEL))
            if not model.is_absolute():
                model = PROJECT_DIR / model
            if not model.is_file():
                messagebox.showerror("模型文件不存在", str(model), parent=self.root)
                return
        self.cfg["auto_hook"] = bool(self.auto_hook_var.get())
        self.hotkey_stopped = False
        self.running = True
        self.worker_error = None
        self.status = "等待游戏中的钓鱼小游戏"
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()
        self.btn.config(text="停止辅助", bg="#cc4d4d", activebackground="#a83c3c")
        self._set_controls(False)

    def stop(self):
        if not self.running:
            return
        self.running = False
        self.status = "正在安全停止…"
        self.btn.config(text="正在停止…", state=tk.DISABLED, bg="#888888")
        self._set_controls(False)

    def _save_auto_hook(self):
        cfg = load_config()
        cfg["auto_hook"] = bool(self.auto_hook_var.get())
        save_config(cfg)
        self.cfg = cfg

    def on_close(self):
        if self.running or (self.thread is not None and self.thread.is_alive()):
            self.closing = True
            if self.running:
                self.stop()
            self.root.after(80, self._close_when_stopped)
            return
        self.root.destroy()

    def _close_when_stopped(self):
        if self.thread is not None and self.thread.is_alive():
            self.root.after(80, self._close_when_stopped)
        else:
            self.root.destroy()

    def calibrate_progress(self):
        self.calibrate(progress=True)

    def calibrate(self, progress=False):
        if self.running:
            messagebox.showwarning("请先停止", "请先停止辅助，再选择钓鱼区域。", parent=self.root)
            return
        if self.calibration_process is not None and self.calibration_process.poll() is None:
            messagebox.showinfo("正在校准", "区域选择窗口已经打开。", parent=self.root)
            return

        messagebox.showinfo(
            "区域选择说明",
            ("先让钓鱼小游戏显示在屏幕上。确定后有 5 秒切回游戏。\n\n"
             "请框选右侧捕获进度的完整细长槽，从空白顶部到底部；"
             "只选内部，不包括木框，也不要只选当前已填充的部分。\n\n"
             "按 Enter 保存；按 Esc 取消。" if progress else
            "先让钓鱼小游戏显示在屏幕上，再继续。\n\n"
            "接下来会打开一个窗口，请只框选小游戏中上下移动的窄长轨道，"
            "选取内部完整可移动高度，不含木框和右侧进度槽。"
            "按 Enter 保存；按 Esc 取消。重新选择轨道后，需重新选择进度槽。"),
            parent=self.root,
        )

        python = Path(sys.executable)
        if python.name.lower() == "pythonw.exe":
            python = python.with_name("python.exe")
        if not python.exists():
            python = Path(sys.executable)

        try:
            self.calibration_process = subprocess.Popen(
                [str(python), str(PROJECT_DIR / "main.py"), "calibrate-progress" if progress else "calibrate"],
                cwd=str(PROJECT_DIR),
                creationflags=getattr(subprocess, "CREATE_NEW_CONSOLE", 0),
            )
            self.status = "区域校准进行中…"
        except Exception as exc:
            self.calibration_process = None
            messagebox.showerror("无法启动校准", str(exc), parent=self.root)

    def open_logs(self):
        log_dir = Path(PROJECT_DIR) / "logs"
        log_dir.mkdir(exist_ok=True)
        try:
            os.startfile(str(log_dir))
        except Exception as exc:
            messagebox.showerror("无法打开日志目录", str(exc), parent=self.root)

    @staticmethod
    def _translate_status(status):
        if status.startswith("미니게임 제어중"):
            found = re.search(r"bar=(\d+)\s+fish=(\d+|None)", status)
            if found:
                fish = "未识别" if found.group(2) == "None" else found.group(2)
                return "自动控制中  |  钓鱼条 %s  鱼 %s" % (found.group(1), fish)
            return "自动控制中：已识别钓鱼小游戏"
        if status.startswith("대기중") or status == "대기중":
            return "待命中：请在游戏里手动抛竿并钩鱼"
        if status == "정지":
            return "已停止"
        if status.startswith("오류 중단:"):
            return "运行出错并停止：" + status.split(":", 1)[1].strip()
        return status

    def _poll(self):
        if self.calibration_process is not None:
            code = self.calibration_process.poll()
            if code is not None:
                self.calibration_process = None
                self.cfg = load_config()
                self._show_progress_region()
                self.status = (
                    "区域选择已结束；如已保存，请重新开始辅助"
                    if code == 0
                    else "区域校准没有完成，请查看校准窗口"
                )

        if self.running:
            if self.thread is not None and not self.thread.is_alive():
                self.running = False
                self.thread = None
                self.btn.config(text="开始辅助", bg="#2e9d64", activebackground="#267c50")
                self._set_controls(True)
                if self.worker_error:
                    shown = "运行出错并停止：%s" % self.worker_error
                    color = "#b33b32"
                else:
                    shown = "F8 已停止" if self.hotkey_stopped else "已停止"
                    color = "#555555"
                self.lbl.config(text=shown, fg=color)
            else:
                self.lbl.config(text=self._translate_status(self.status), fg="#222222")
        else:
            if self.thread is not None:
                if self.thread.is_alive():
                    self.lbl.config(text="正在安全停止…", fg="#8a5b00")
                else:
                    self.thread = None
                    self.btn.config(
                        text="开始辅助", state=tk.NORMAL,
                        bg="#2e9d64", activebackground="#267c50",
                    )
                    self._set_controls(True)
            shown = "F8 已停止" if self.hotkey_stopped else self._translate_status(self.status)
            if self.thread is None:
                self.lbl.config(text=shown, fg="#555555")

        if not self.closing:
            self.root.after(120, self._poll)


if __name__ == "__main__":
    # pythonw has no console; retain startup and Tk callback diagnostics.
    startup_dir = PROJECT_DIR / "logs"
    startup_dir.mkdir(exist_ok=True)
    with (startup_dir / "assistant_startup.log").open("a", encoding="utf-8", buffering=1) as startup_log:
        sys.stdout = sys.stderr = startup_log
        try:
            app = ChineseAssistGUI()
            app.root.report_callback_exception = lambda kind, value, tb: traceback.print_exception(kind, value, tb)
            print(f"GUI_READY pid={os.getpid()} python={sys.executable} controller={app.cfg.get('controller', 'ppo')}", flush=True)
            app.run()
        except Exception:
            traceback.print_exc()
            raise
