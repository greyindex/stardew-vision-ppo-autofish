"""Independent Chinese desktop GUI for the visual CNN + frozen PPO."""
import ctypes
from ctypes import wintypes
from datetime import datetime
import json
import os
from pathlib import Path
import queue
import sys
import tkinter as tk
from tkinter import ttk, messagebox
import traceback

from PIL import Image, ImageDraw, ImageTk
import cv2
import mss
import numpy as np
from pynput import keyboard

from vision_live import ROOT, VisionSession, Windows, enable_dpi_awareness, load_settings, save_settings

LEVELS = ["按绿条长度估计", *map(str, range(21))]
TACKLES = {"无渔具": "none", "软木塞浮标": "cork", "铅制浮标": "lead",
           "陷阱浮标": "trap", "倒刺钩": "barbed"}
MODES = {"视觉 + PPO 控制": "control", "只观察（不操作鼠标）": "observe"}
BITE_MODES = {"音效或全屏 !（推荐）": "audio_visual", "仅音效指纹识别": "audio",
              "全屏 ! 确认 + 音效辅助": "visual_audio", "仅全屏 !（无需声音）": "visual",
              "鱼竿自动上钩附魔": "enchanted"}


class PanelSelection(tk.Toplevel):
    def __init__(self, parent, image, client, on_selected):
        super().__init__(parent)
        self.title("框选完整钓鱼面板")
        self.client, self.on_selected = client, on_selected
        self.origin, self.box = None, None
        max_w = min(self.winfo_screenwidth() - 100, 1400)
        max_h = min(self.winfo_screenheight() - 180, 850)
        self.scale = min(max_w / image.shape[1], max_h / image.shape[0], 1.)
        size = (round(image.shape[1] * self.scale), round(image.shape[0] * self.scale))
        tk.Label(self, text="拖框整个钓鱼面板：包含左侧白色外框、鱼道、右侧进度槽与底部。Enter 确认；Esc 取消。",
                 font=("Microsoft YaHei UI", 10)).pack(padx=12, pady=10)
        self.photo = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).resize(size))
        self.canvas = tk.Canvas(self, width=size[0], height=size[1], highlightthickness=0, cursor="crosshair")
        self.canvas.pack(padx=12)
        self.canvas.create_image(0, 0, image=self.photo, anchor="nw")
        ttk.Button(self, text="使用此面板选区", command=self.confirm).pack(pady=10)
        self.canvas.bind("<ButtonPress-1>", self.begin)
        self.canvas.bind("<B1-Motion>", self.drag)
        self.bind("<Return>", lambda event: self.confirm())
        self.bind("<Escape>", lambda event: self.destroy())
        self.grab_set()
        self.focus_force()

    def begin(self, event):
        self.origin = (event.x, event.y)
        if self.box:
            self.canvas.delete(self.box)
        self.box = self.canvas.create_rectangle(event.x, event.y, event.x, event.y,
                                                 outline="#00e4d4", width=2)

    def drag(self, event):
        if self.origin:
            x = max(0, min(event.x, int(self.canvas["width"])))
            y = max(0, min(event.y, int(self.canvas["height"])))
            self.canvas.coords(self.box, *self.origin, x, y)

    def confirm(self):
        if not self.box:
            return
        x0, y0, x1, y1 = self.canvas.coords(self.box)
        x0, x1 = sorted((x0 / self.scale, x1 / self.scale))
        y0, y1 = sorted((y0 / self.scale, y1 / self.scale))
        w, h = round(x1 - x0), round(y1 - y0)
        if h < 150 or w < 40 or not .24 <= w / h <= .40:
            messagebox.showinfo("选区范围", "请框选整个细长面板，长宽比约为 300:94。", parent=self)
            return
        self.on_selected({"roi": {"left": round(x0), "top": round(y0), "width": w, "height": h},
                          "client_size": [self.client["width"], self.client["height"]]})
        self.destroy()


class VisionGUI:
    def __init__(self):
        enable_dpi_awareness()
        self.root = tk.Tk()
        self.root.title("星露谷视觉助手 · CNN + PPO（独立版）")
        self.root.geometry("930x880+90+40")
        self.root.minsize(870, 800)
        self.root.configure(bg="#f3f6fa")
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.config = load_settings()
        self.closing = False
        self.keys = queue.SimpleQueue()
        self.hotkeys_down = set()
        self.schedule_generation = 0
        self.pending_auto = False
        self.last_preview = None
        self.selector = None
        self.controls = []
        self.build()
        self.root.update_idletasks()
        wanted_w = min(max(930, self.root.winfo_reqwidth()), self.root.winfo_screenwidth()-100)
        wanted_h = min(max(880, self.root.winfo_reqheight()), self.root.winfo_screenheight()-100)
        self.root.geometry(f"{wanted_w}x{wanted_h}+60+40")
        self.engine = VisionSession(self.config)
        self.listener = keyboard.Listener(on_press=self.on_key, on_release=self.on_key_release)
        self.listener.start()
        self.root.after(50, self.poll)

    def build(self):
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure("TFrame", background="#f3f6fa")
        style.configure("TLabel", background="#f3f6fa", font=("Microsoft YaHei UI", 10))
        style.configure("TButton", font=("Microsoft YaHei UI", 10), padding=7)
        style.configure("TCombobox", padding=4)
        style.configure("TLabelframe", background="#f3f6fa")
        style.configure("TLabelframe.Label", background="#f3f6fa", font=("Microsoft YaHei UI", 10, "bold"))
        outer = ttk.Frame(self.root, padding=16)
        outer.pack(fill="both", expand=True)
        outer.columnconfigure(0, weight=1)
        outer.columnconfigure(1, weight=0)
        outer.rowconfigure(2, weight=1)
        ttk.Label(outer, text="视觉识别 × PPO", font=("Microsoft YaHei UI", 19, "bold")).grid(row=0, column=0, sticky="w")
        ttk.Label(outer, text="真实游戏实测", foreground="#64748b").grid(row=0, column=1, sticky="e")
        self.model_text = tk.StringVar(value="正在加载视觉模型与 1000 万步 PPO…")
        ttk.Label(outer, textvariable=self.model_text, foreground="#64748b", wraplength=830).grid(
            row=1, column=0, columnspan=2, sticky="w", pady=(5, 15))
        left = ttk.Frame(outer)
        left.grid(row=2, column=0, sticky="nsew", padx=(0, 18))
        left.columnconfigure(0, weight=1)
        right = ttk.Frame(outer)
        right.grid(row=2, column=1, sticky="nsew")

        launch_buttons = ttk.Frame(left)
        launch_buttons.grid(row=0, column=0, sticky="ew")
        self.start_button = tk.Button(launch_buttons, text="正在加载…", font=("Microsoft YaHei UI", 13, "bold"),
                                      bg="#187e69", fg="white", activebackground="#126353", activeforeground="white",
                                      relief="flat", height=2, command=self.toggle, state="disabled")
        self.start_button.pack(side="left", fill="x", expand=True, padx=(0, 4))
        self.full_auto_button = tk.Button(launch_buttons, text="F1 全自动", font=("Microsoft YaHei UI", 13, "bold"),
                                         bg="#3864a2", fg="white", activebackground="#2a5188", activeforeground="white",
                                         relief="flat", height=2, command=self.delayed_auto, state="disabled")
        self.full_auto_button.pack(side="left", fill="x", expand=True, padx=(4, 0))
        self.status_text = tk.StringVar(value="初始化中；尚未采集或发送输入")
        ttk.Label(left, textvariable=self.status_text, wraplength=485, foreground="#215469").grid(
            row=1, column=0, sticky="w", pady=(9, 12))

        settings = ttk.LabelFrame(left, text="本次实测设置", padding=10)
        settings.grid(row=2, column=0, sticky="ew")
        settings.columnconfigure(1, weight=1)
        self.mode = tk.StringVar(value=next(k for k, v in MODES.items() if v == self.config["mode"]))
        self.level = tk.StringVar(value="按绿条长度估计" if self.config["level"] == "auto" else str(self.config["level"]))
        self.tackle = tk.StringVar(value=next(k for k, v in TACKLES.items() if v == self.config["tackle"]))
        for row, (title, variable, choices) in enumerate((("模式", self.mode, list(MODES)),
                    ("有效钓鱼等级", self.level, LEVELS), ("渔具", self.tackle, list(TACKLES)))):
            ttk.Label(settings, text=title).grid(row=row, column=0, sticky="w", pady=4)
            widget = ttk.Combobox(settings, textvariable=variable, values=choices, state="readonly", width=27)
            widget.grid(row=row, column=1, sticky="ew", padx=(8, 0), pady=4)
            self.controls.append((widget, "readonly"))
        self.offset = tk.StringVar(value=str(self.config["fish_offset_native"]))
        ttk.Label(settings, text="鱼中心坐标校正").grid(row=3, column=0, sticky="w", pady=4)
        offset_row = ttk.Frame(settings)
        offset_row.grid(row=3, column=1, sticky="ew", padx=(8, 0))
        widget = ttk.Spinbox(offset_row, from_=-10, to=15, increment=.5, textvariable=self.offset, width=7)
        widget.pack(side="left")
        self.controls.append((widget, "normal"))
        ttk.Label(offset_row, text="原图 px（实机校准 +1）", foreground="#64748b", font=("Microsoft YaHei UI", 9)).pack(side="left", padx=8)
        self.bite = tk.StringVar(value=next(k for k, v in BITE_MODES.items()
                                          if v == self.config.get("auto_bite", "audio_visual")))
        ttk.Label(settings, text="全自动上钩方式").grid(row=4, column=0, sticky="w", pady=4)
        widget = ttk.Combobox(settings, textvariable=self.bite, values=list(BITE_MODES), state="readonly", width=27)
        widget.grid(row=4, column=1, sticky="ew", padx=(8, 0), pady=4)
        self.controls.append((widget, "readonly"))
        ttk.Label(settings, text="等级可包含食物加成。PPO 首轮训练覆盖 Lv.0–10、无渔具。",
                  foreground="#64748b", font=("Microsoft YaHei UI", 9), wraplength=460).grid(
                      row=5, column=0, columnspan=2, sticky="w", pady=(5, 0))

        locate = ttk.Frame(left)
        locate.grid(row=3, column=0, sticky="ew", pady=(10, 3))
        button = ttk.Button(locate, text="恢复自动定位", command=self.auto_region)
        button.pack(side="left", fill="x", expand=True, padx=(0, 4))
        self.controls.append((button, "normal"))
        button = ttk.Button(locate, text="3 秒后框选面板", command=self.delayed_select)
        button.pack(side="left", fill="x", expand=True, padx=(4, 0))
        self.controls.append((button, "normal"))
        self.region_text = tk.StringVar()
        self.update_region_label()
        ttk.Label(left, textvariable=self.region_text, wraplength=480, foreground="#64748b",
                  font=("Microsoft YaHei UI", 9)).grid(row=4, column=0, sticky="w", pady=(2, 9))

        telemetry = ttk.LabelFrame(left, text="实时观测", padding=10)
        telemetry.grid(row=5, column=0, sticky="ew")
        self.action_text = tk.StringVar(value="鼠标：已松开")
        ttk.Label(telemetry, textvariable=self.action_text, font=("Microsoft YaHei UI", 12, "bold")).pack(anchor="w")
        self.auto_text = tk.StringVar(value="F1：在水边选好鱼竿、鼠标指向水面后开始循环")
        ttk.Label(telemetry, textvariable=self.auto_text, foreground="#3864a2", wraplength=480,
                  font=("Microsoft YaHei UI", 9)).pack(anchor="w", pady=(4, 0))
        self.geometry_text = tk.StringVar(value="等待面板、鱼、绿条和捕获进度")
        ttk.Label(telemetry, textvariable=self.geometry_text, wraplength=465).pack(anchor="w", pady=(7, 4))
        self.timing_text = tk.StringVar(value="30 Hz 决策 · 采集耗时待测")
        ttk.Label(telemetry, textvariable=self.timing_text, wraplength=465).pack(anchor="w")
        self.score_text = tk.StringVar(value="存在分数：尚无观测（分数未经校准）")
        ttk.Label(telemetry, textvariable=self.score_text, font=("Microsoft YaHei UI", 9),
                  foreground="#64748b", wraplength=465).pack(anchor="w", pady=(5, 0))

        buttons = ttk.Frame(left)
        buttons.grid(row=6, column=0, sticky="ew", pady=10)
        ttk.Button(buttons, text="F9 保存诊断片段", command=lambda: self.engine.request_diagnostic()).pack(side="left", fill="x", expand=True, padx=(0, 4))
        ttk.Button(buttons, text="打开日志目录", command=self.open_logs).pack(side="left", padx=4)
        self.log_box = tk.Text(left, height=5, wrap="word", font=("Microsoft YaHei UI", 9),
                               bg="#e9eef5", fg="#334155", relief="flat", state="disabled")
        self.log_box.grid(row=7, column=0, sticky="nsew")
        left.rowconfigure(7, weight=1)

        ttk.Label(right, text="面板 / 头顶咬钩区域", font=("Microsoft YaHei UI", 11, "bold")).pack(anchor="w")
        self.preview_label = tk.Label(right, text="小游戏出现后显示识别叠图\n\n可将此窗口放在另一块屏幕",
                                      bg="#14202f", fg="#cbd5e1", font=("Microsoft YaHei UI", 10))
        self.preview_label.pack(fill="both", expand=True, pady=(8, 6))
        self.placeholder = ImageTk.PhotoImage(Image.new("RGB", (235, 590), "#14202f"))
        self.preview_label.configure(image=self.placeholder, compound="center")
        ttk.Label(right, text="青：鱼中心　白：PPO 输入中心\n紫：绿条　橙：进度 / 咬钩搜索区域\n等待时：绿框标出新出现的 !",
                  font=("Microsoft YaHei UI", 8), foreground="#64748b").pack(anchor="w")

        ttk.Label(outer, text="先按 F8 停止旧助手。水边站好、选鱼竿、鼠标指水面后按 F1，全自动循环；再次 F1 或 F8 停止。\n"
                             "普通开始：手动抛竿上钩。F6 框选 · F9 保存诊断。全自动中切出游戏或移动/打开菜单会停止。",
                  font=("Microsoft YaHei UI", 9), foreground="#64748b", wraplength=850).grid(
                      row=3, column=0, columnspan=2, sticky="w", pady=(13, 0))

    def read_config(self):
        offset = float(self.offset.get())
        if not -10 <= offset <= 15:
            raise ValueError("鱼中心校正应在 -10 到 +15 原图像素之间")
        return {**self.config, "mode": MODES[self.mode.get()],
                "level": "auto" if self.level.get() == LEVELS[0] else int(self.level.get()),
                "tackle": TACKLES[self.tackle.get()], "fish_offset_native": offset,
                "calibration_profile": "real-game-20260927-v1" if offset == 1.0 else "manual-offset",
                "auto_bite": BITE_MODES[self.bite.get()], "full_auto": False}

    def toggle(self):
        self.cancel_pending()
        if self.engine.active.is_set():
            self.engine.stop()
            return
        try:
            self.config = self.read_config()
            save_settings(self.config)
            self.engine.start(self.config)
        except Exception as exc:
            messagebox.showerror("设置有误", str(exc), parent=self.root)

    def delayed_auto(self):
        self.cancel_pending()
        if self.engine.active.is_set() and self.engine.config.get("full_auto"):
            self.engine.stop("全自动已停止 · 输入已释放")
            return
        self.engine.stop("3 秒内切回游戏，站在水边并把鼠标指向投点")
        self.pending_auto = True
        token = self.schedule_generation
        self.root.iconify()
        self.root.after(3000, lambda: self.start_auto(token=token))

    def cancel_pending(self):
        self.schedule_generation += 1
        self.pending_auto = False

    def start_auto(self, point=None, token=None):
        if self.closing or (token is not None and token != self.schedule_generation):
            return
        self.pending_auto = False
        if not self.engine.ready.is_set():
            self.add_log("模型尚未就绪，请稍后再按 F1。")
            return
        try:
            windows = self.engine.windows
            hwnd = windows.find_game()
            client = windows.client(hwnd)
            if not client or not windows.focused(hwnd):
                raise ValueError("请在星露谷前台、水边站好并选中鱼竿后按 F1。")
            if point is None:
                cursor = wintypes.POINT()
                if not windows.user.GetCursorPos(ctypes.byref(cursor)):
                    raise ValueError("无法读取鼠标投点")
                point = (cursor.x, cursor.y)
            x, y = point[0]-client["left"], point[1]-client["top"]
            if not (0 <= x < client["width"] and 0 <= y < client["height"]):
                raise ValueError("鼠标投点必须位于游戏窗口里的水面。")
            self.mode.set("视觉 + PPO 控制")
            self.config = self.read_config()
            manual = self.config.get("manual_panel")
            if manual and manual["client_size"] != [client["width"], client["height"]]:
                self.config["manual_panel"] = None
                self.update_region_label()
                self.add_log("窗口尺寸已改变，本次已恢复自动面板定位。")
            save_settings(self.config)
            runtime = {**self.config, "full_auto": True,
                       "auto_origin": {"x": x, "y": y, "client_size": [client["width"], client["height"]]}}
            if token is not None and token != self.schedule_generation:
                return
            self.engine.stop("切换到全自动")
            self.engine.start(runtime)
            self.add_log(f"F1 全自动已启动，记住窗口内投点 ({x}, {y})。")
        except Exception as exc:
            self.engine.stop(str(exc))
            self.add_log(str(exc))
            self.root.deiconify()

    def update_region_label(self):
        panel = self.config.get("manual_panel")
        self.region_text.set("定位：自动搜索完整面板；找不到时可在游戏内按 F6 框选。" if not panel else
                             f"定位：手动完整面板 {panel['roi']['width']} × {panel['roi']['height']}；坐标随游戏窗口移动。")

    def auto_region(self):
        self.cancel_pending()
        self.engine.stop()
        self.config["manual_panel"] = None
        save_settings(self.config)
        self.update_region_label()

    def delayed_select(self):
        self.cancel_pending()
        token = self.schedule_generation
        self.engine.stop("请切回游戏，3 秒后采集框选画面")
        self.root.iconify()
        self.root.after(3000, lambda: self.select_panel() if token == self.schedule_generation else None)

    def select_panel(self):
        if self.closing:
            return
        if self.selector is not None and self.selector.winfo_exists():
            self.selector.lift()
            return
        self.engine.stop("框选完成后请重新点击开始")
        try:
            windows = self.engine.windows
            hwnd = windows.find_game()
            client = windows.client(hwnd)
            if not client or not windows.focused(hwnd):
                raise RuntimeError("请在钓鱼小游戏出现时按 F6，或点击框选按钮后于 3 秒内切回游戏。")
            with mss.mss() as sct:
                image, _ = VisionSession._capture(sct, client)
            self.root.deiconify()
            self.selector = PanelSelection(self.root, image, client, self.selected)
        except Exception as exc:
            self.root.deiconify()
            messagebox.showinfo("面板框选", str(exc), parent=self.root)

    def selected(self, panel):
        self.config["manual_panel"] = panel
        save_settings(self.config)
        self.update_region_label()
        self.add_log("完整面板选区已保存；点击开始后切回游戏。")

    def on_key(self, key):
        if key in (keyboard.Key.f1, keyboard.Key.f6, keyboard.Key.f8, keyboard.Key.f9):
            if key in self.hotkeys_down:
                return
            self.hotkeys_down.add(key)
        if key == keyboard.Key.f8:
            self.cancel_pending()
            self.engine.stop("F8 已停止 · 输入已释放")
        elif key == keyboard.Key.f9:
            self.engine.request_diagnostic()
        elif key == keyboard.Key.f6:
            self.cancel_pending()
            self.engine.stop("正在框选面板")
            self.keys.put("select")
        elif key == keyboard.Key.f1:
            if self.pending_auto or (self.engine.active.is_set() and self.engine.config.get("full_auto")):
                self.cancel_pending()
                self.engine.stop("F1 已停止全自动 · 输入已释放")
            else:
                self.cancel_pending()
                self.pending_auto = True
                cursor = wintypes.POINT()
                if self.engine.windows.user.GetCursorPos(ctypes.byref(cursor)):
                    self.keys.put(("auto", (cursor.x, cursor.y), self.schedule_generation))
                else:
                    self.pending_auto = False
        elif self.engine.active.is_set() and self.engine.config.get("full_auto"):
            char = getattr(key, "char", None)
            manual = key in (keyboard.Key.esc, keyboard.Key.tab, keyboard.Key.enter,
                             keyboard.Key.up, keyboard.Key.down, keyboard.Key.left, keyboard.Key.right)
            manual = manual or (char is not None and char.lower() in "wasdeit0123456789 ")
            if manual:
                self.engine.stop("全自动已停止：检测到移动/工具切换/菜单按键；站好后可重新按 F1")

    def on_key_release(self, key):
        self.hotkeys_down.discard(key)

    def add_log(self, message):
        self.log_box.configure(state="normal")
        self.log_box.insert("end", datetime.now().strftime("%H:%M:%S ") + message + "\n")
        if int(self.log_box.index("end-1c").split(".")[0]) > 250:
            self.log_box.delete("1.0", "100.0")
        self.log_box.see("end")
        self.log_box.configure(state="disabled")

    def open_logs(self):
        folder = ROOT / "logs"
        folder.mkdir(exist_ok=True)
        os.startfile(str(folder))

    def draw_preview(self, item):
        if item is self.last_preview:
            return
        self.last_preview = item
        crop, record = item
        image = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        d = ImageDraw.Draw(image)
        sx, sy = image.width / 94, image.height / 300
        colors = (("fish_visual_center", "#00ffff", 30, 55), ("bar_top", "#f36bff", 30, 55),
                  ("bar_bottom", "#f36bff", 30, 55), ("progress_top", "#ffad36", 61, 74))
        if record.get("type") == "bite_frame":
            evidence = record["evidence"]
            for name, color in (("excluded_region", "#e85454"), ("region", "#ffad36"),
                                ("stem", "#4dff8b"), ("dot", "#4dff8b")):
                box = evidence.get(name)
                if box:
                    x, y, w, h = box
                    d.rectangle((x, y, x+w, y+h), outline=color, width=2)
        # Invalid rows are deliberately not drawn as if they were accepted input.
        elif not record["reason"]:
            for name, color, x0, x1 in colors:
                y = record["native_rows"][name] * sy
                d.line((x0 * sx, y, x1 * sx, y), fill=color, width=max(1, round(sy)))
            y = record["geometry"]["fish_center"] * sy
            d.line((50 * sx, y, 56 * sx, y), fill="white", width=max(1, round(sy)))
        else:
            d.rectangle((0, 0, image.width - 1, image.height - 1), outline="#e89d3d", width=3)
        max_h = max(350, self.preview_label.winfo_height() - 4)
        max_w = max(200, self.preview_label.winfo_width() - 4)
        scale = min(max_w / image.width, max_h / image.height)
        image = image.resize((round(image.width * scale), round(image.height * scale)), Image.Resampling.NEAREST)
        self.photo = ImageTk.PhotoImage(image)
        self.preview_label.configure(image=self.photo, text="")

    def poll(self):
        if self.closing:
            return
        while not self.keys.empty():
            command = self.keys.get()
            if command == "select":
                self.select_panel()
            elif isinstance(command, tuple) and command[0] == "auto":
                self.start_auto(command[1], token=command[2])
        while not self.engine.messages.empty():
            self.add_log(self.engine.messages.get())
        state, preview = self.engine.snapshot()
        running = self.engine.active.is_set()
        ready = bool(state.get("ready"))
        self.start_button.configure(text="停止 · F8" if running else ("小游戏辅助" if ready else "加载中…" if not state.get("error") else "加载失败"),
                                    bg="#c34a50" if running else "#187e69", state="normal" if ready else "disabled")
        full_auto = running and self.engine.config.get("full_auto")
        self.full_auto_button.configure(text="停止全自动 · F1" if full_auto else "F1 全自动",
                                        state="normal" if ready else "disabled", bg="#c34a50" if full_auto else "#3864a2")
        for widget, idle_state in self.controls:
            widget.configure(state="disabled" if running else idle_state)
        self.status_text.set(state["status"])
        if "models" in state:
            m = state["models"]
            self.model_text.set(f"视觉 {m['vision_step']:,} 步 · PPO {m['ppo_steps']:,} 步 · {m['gpu']}")
        self.action_text.set("鼠标：" + ("按住" if self.engine.mouse.holding else "已松开") +
                             (f"　PPO 建议：{'按住' if state.get('suggested') else '松开'}"
                              if running and state.get("focused") and (not full_auto or state.get("auto_phase") == "playing") else ""))
        if full_auto:
            from auto_fishing import AutoFishingCycle
            phase = AutoFishingCycle.LABELS.get(state.get("auto_phase"), "准备")
            energy = state.get("auto_energy")
            energy_text = f"{energy:.0%}" if energy is not None else "未读到"
            duration = state.get("auto_hold_seconds")
            power_text = (f" · 按住 {state.get('auto_hold_elapsed', 0):.2f}/{duration:.2f}s"
                          if duration is not None and state.get("auto_phase") == "charge" else "")
            bite_text = ""
            if state.get("auto_phase") in ("settle", "wait"):
                audio = state.get("bite_audio")
                bite_mode = self.engine.config.get("auto_bite")
                audio_text = ("音频已连接" if audio and audio.get("ok") else
                              "音频连接中" if audio is not None else "纯视觉/附魔")
                if bite_mode in ("audio", "audio_visual"):
                    bite_text = (f"\n{audio_text} · 音效匹配 {(audio or {}).get('score', 0):.2f}"
                                 f" / {((audio or {}).get('threshold') or .93):.2f}"
                                 f" · 音量 {(audio or {}).get('rms', 0):.3f}"
                                 + (f" · 全屏 ! {state.get('bite_visual_frames', 0)} 帧" if bite_mode == "audio_visual"
                                    else " · 无需感叹号"))
                else:
                    bite_text = (f"\n全屏检测 {state.get('bite_hz', 0):.1f} Hz · ! 连续 {state.get('bite_visual_frames', 0)} 帧"
                                 f" · {audio_text}")
            self.auto_text.set(f"全自动第 {state.get('auto_cast', 0)} 杆 · {phase}{power_text} · 精力 {energy_text}{bite_text}")
        else:
            self.auto_text.set("F1：水边站好、选中鱼竿、鼠标指向水面后开始循环")
        if "geometry" in state:
            g = state["geometry"]
            level_source = "（按条长估计）" if self.engine.config.get("level") == "auto" else "（手动）"
            self.geometry_text.set(f"第 {state.get('episode', 0)} 局 · Lv.{state.get('level', 0)}{level_source} · 进度 {g['progress']:.0%}\n"
                                   f"鱼中心 {g['fish_center']:.1f}　绿条 {g['bar_top']:.1f}–{g['bar_bottom']:.1f}（原图坐标）")
            self.timing_text.set(f"实测 {state.get('hz', 0):.1f} / 目标 30 Hz · 截图 {state.get('capture_ms', 0):.1f} ms\n"
                                 f"CNN {state.get('cnn_ms', 0):.1f} · PPO {state.get('ppo_ms', 0):.1f} ms"
                                 f" · 图像到输入 {state.get('age_ms', 0):.1f} ms")
            p = state.get("scores", {})
            evidence = state.get("evidence", {})
            fish_source = "鱼图标素材" if evidence.get("fish_source") == "sprite" else "CNN"
            self.score_text.set(f"存在分数：面板 {p.get('panel', 0):.2f}　鱼 {p.get('fish', 0):.2f}　"
                                 f"绿条 {p.get('bar', 0):.2f}　进度 {p.get('progress', 0):.2f}\n"
                                 f"鱼定位：{fish_source} · 素材匹配 {evidence.get('fish_template_score', 0):.2f}"
                                 f" · 宝箱遮挡 {'是' if evidence.get('treasure_near_fish') else '否'}")
        if full_auto and state.get("auto_phase") in ("settle", "wait"):
            if self.engine.config.get("auto_bite") == "audio_visual":
                self.timing_text.set("音效指纹或连续两帧全屏 ! 均可独立上钩。\n"
                                     "已排除右上角任务区；无需先定位玩家。")
            elif self.engine.config.get("auto_bite") == "audio":
                self.timing_text.set("音效指纹独立确认上钩；只在等待阶段启用。\n"
                                     "保留游戏音效音量；音乐、环境音无需关闭。")
            else:
                self.timing_text.set(f"全屏检测：实测 {state.get('bite_hz', 0):.1f} Hz\n"
                                     "提竿需要新出现的 !；已排除右上角任务区。")
        if preview is not None:
            self.draw_preview(preview)
        self.root.after(50, self.poll)

    def close(self):
        if self.closing:
            return
        self.closing = True
        self.cancel_pending()
        self.engine.close()
        self.listener.stop()
        self.root.withdraw()
        self.await_exit()

    def await_exit(self):
        if self.engine.worker.is_alive():
            self.root.after(50, self.await_exit)
        else:
            self.root.destroy()

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    # Prevent two new GUI instances from pressing/releasing against one another.
    kernel = ctypes.windll.kernel32
    kernel.CreateMutexW.argtypes = [ctypes.c_void_p, ctypes.c_bool, ctypes.c_wchar_p]
    kernel.CreateMutexW.restype = ctypes.c_void_p
    handle = kernel.CreateMutexW(None, False, "Local\\StardewVisionPPOGUI")
    if kernel.GetLastError() == 183:
        ctypes.windll.user32.MessageBoxW(None, "视觉助手已经打开。\n如果刚更新了程序，请先关闭现有窗口，再重新启动以加载新版。", "视觉助手", 0)
        sys.exit(0)
    try:
        VisionGUI().run()
    except Exception:
        error = traceback.format_exc()
        folder = ROOT / "logs"
        folder.mkdir(exist_ok=True)
        path = folder / "vision_gui_startup_error.log"
        path.write_text(error, encoding="utf-8")
        ctypes.windll.user32.MessageBoxW(None, "启动失败，错误已保存到：\n" + str(path), "视觉助手", 0x10)
        raise
