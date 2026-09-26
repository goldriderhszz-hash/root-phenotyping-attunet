"""RootScope native desktop application. No browser or local web server is used."""
from __future__ import annotations

import os
import json
import queue
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageTk

try:
    from tkinterdnd2 import DND_FILES, TkinterDnD
    Window = TkinterDnD.Tk
except ImportError:
    DND_FILES = None
    Window = tk.Tk

from rootscope.batch import IMAGE_SUFFIXES, run_batch
from rootscope import __version__
from rootscope.engine import DEFAULT_THRESHOLD
from rootscope.models import DEFAULT_MODEL_ID, catalog, model_spec

BG = "#f4f8f6"
WHITE = "#ffffff"
INK = "#193544"
MUTED = "#698085"
NAVY = "#113841"
TEAL = "#147e78"
PALE = "#e7f4ef"
LINE = "#dfe9e5"
AMBER = "#b76d3b"
FONT = "Microsoft YaHei UI" if sys.platform == "win32" else "DejaVu Sans"
FILE_TYPES = [("图像文件", "*.tif *.tiff *.png *.jpg *.jpeg *.bmp"), ("所有文件", "*.*")]
LAYERS = {"原图": None, "叠加": "overlay.png", "掩膜": "prediction_mask.png", "骨架": "skeleton.png"}


def label(parent, text, size=10, weight="normal", color=INK, bg=WHITE, **kwargs):
    return tk.Label(parent, text=text, font=(FONT, size, weight), fg=color, bg=bg, **kwargs)


def button(parent, text, command, primary=False, width=None):
    return tk.Button(parent, text=text, command=command, relief="flat", bd=0,
                     font=(FONT, 10, "bold"), padx=16, pady=9, width=width,
                     cursor="hand2", activeforeground=WHITE if primary else TEAL,
                     activebackground="#106b66" if primary else "#eff7f3",
                     fg=WHITE if primary else TEAL, bg=TEAL if primary else PALE)


class RootScopeApp(Window):
    def __init__(self):
        super().__init__()
        self.title("RootScope | 根系图像表型分析")
        self.geometry("1460x960")
        self.minsize(1080, 700)
        self.configure(bg=BG)
        self.images: list[Path] = []
        self.references: list[Path] = []
        self.rows: dict[str, dict] = {}
        self.current_result: dict | None = None
        self.preview_image = None
        self.preview_after = None
        self.events: queue.Queue = queue.Queue()
        self.cancel_event = threading.Event()
        self.running = False
        self.pending_close = False
        self.last_output: Path | None = None
        self.last_archive: Path | None = None
        self.threshold = tk.DoubleVar(value=DEFAULT_THRESHOLD)
        self.model_id = tk.StringVar(value=DEFAULT_MODEL_ID)
        self.save_probability = tk.BooleanVar(value=False)
        self.output_folder = tk.StringVar(value=str(Path.home() / "Documents" / "RootScope Results"))
        self.layer = tk.StringVar(value="叠加")
        self._style()
        self._layout()
        self._shortcuts()
        self.after(100, self._poll)
        self.protocol("WM_DELETE_WINDOW", self._close)

    def _style(self):
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("Root.Treeview", font=(FONT, 10), rowheight=31, background=WHITE,
                        fieldbackground=WHITE, foreground=INK, borderwidth=0)
        style.configure("Root.Treeview.Heading", font=(FONT, 9, "bold"), background="#eef4f1",
                        foreground="#567176", padding=8, relief="flat")
        style.map("Root.Treeview", background=[("selected", PALE)], foreground=[("selected", INK)])
        style.configure("Root.Horizontal.TProgressbar", troughcolor="#e0eee8", background=TEAL,
                        bordercolor="#e0eee8", lightcolor=TEAL, darkcolor=TEAL)
        style.configure("Root.TCheckbutton", background=WHITE, foreground=INK, font=(FONT, 9))

    def _layout(self):
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        sidebar = tk.Frame(self, bg=NAVY, width=226)
        sidebar.grid(row=0, column=0, sticky="ns")
        sidebar.grid_propagate(False)
        label(sidebar, "◈  RootScope", 18, "bold", WHITE, NAVY).pack(anchor="w", padx=25, pady=(31, 2))
        label(sidebar, "RESEARCH DESKTOP", 8, "bold", "#9ebcb8", NAVY).pack(anchor="w", padx=54)
        label(sidebar, "工作空间", 9, "bold", "#8aadaa", NAVY).pack(anchor="w", padx=24, pady=(58, 13))
        self.analysis_nav = tk.Button(sidebar, text="▦   图像分析", command=lambda: self._show_page("analysis"),
                                      anchor="w", padx=20, pady=12, bd=0, relief="flat",
                                      font=(FONT, 11, "bold"), fg=WHITE, bg="#24565b", cursor="hand2")
        self.analysis_nav.pack(fill="x", padx=13, pady=3)
        self.methods_nav = tk.Button(sidebar, text="◇   方法与证据", command=lambda: self._show_page("methods"),
                                     anchor="w", padx=20, pady=12, bd=0, relief="flat",
                                     font=(FONT, 11), fg="#c1d9d5", bg=NAVY, cursor="hand2")
        self.methods_nav.pack(fill="x", padx=13, pady=3)
        tk.Frame(sidebar, bg="#2e5c5e", height=1).pack(fill="x", padx=25, pady=30)
        label(sidebar, "研究模型", 9, "bold", "#8aadaa", NAVY).pack(anchor="w", padx=24)
        label(sidebar, "●  Attention U-Net", 10, "bold", "#e2f3ed", NAVY).pack(anchor="w", padx=24, pady=(16, 3))
        self.model_sidebar = label(sidebar, "固定骨架损失 · Seed 42 · Fold 0", 8, color="#a9c8c3", bg=NAVY)
        self.model_sidebar.pack(anchor="w", padx=41)
        label(sidebar, "本地桌面分析 · 无需浏览器", 8, color="#a6c6c0", bg=NAVY).pack(side="bottom", anchor="w", padx=24, pady=28)

        outer = tk.Frame(self, bg=BG)
        outer.grid(row=0, column=1, sticky="nsew")
        outer.grid_rowconfigure(1, weight=1)
        outer.grid_columnconfigure(0, weight=1)
        topbar = tk.Frame(outer, bg=WHITE, height=59, highlightbackground=LINE, highlightthickness=1)
        topbar.grid(row=0, column=0, sticky="ew")
        topbar.grid_propagate(False)
        self.breadcrumb = label(topbar, "RootScope  /  图像分析", 9, color=MUTED, bg=WHITE)
        self.breadcrumb.pack(side="left", padx=32)
        label(topbar, f"研究版 {__version__}   ·   ONNX 本地推理", 9, color=TEAL, bg=WHITE).pack(side="right", padx=32)
        self.pages = tk.Frame(outer, bg=BG)
        self.pages.grid(row=1, column=0, sticky="nsew")
        self.pages.grid_rowconfigure(0, weight=1)
        self.pages.grid_columnconfigure(0, weight=1)
        self.analysis_page = tk.Frame(self.pages, bg=BG)
        self.methods_page = tk.Frame(self.pages, bg=BG)
        self.analysis_page.grid(row=0, column=0, sticky="nsew")
        self.methods_page.grid(row=0, column=0, sticky="nsew")
        self._analysis_layout()
        self._methods_layout()
        self._show_page("analysis")
        footer = tk.Frame(outer, bg=WHITE, height=54, highlightbackground=LINE, highlightthickness=1)
        footer.grid(row=2, column=0, sticky="ew")
        footer.grid_columnconfigure(0, weight=1)
        self.status = label(footer, "就绪。导入原图后即可开始分析。", 9, color=MUTED, bg=WHITE)
        self.status.grid(row=0, column=0, sticky="w", padx=25, pady=15)
        self.open_output_button = button(footer, "打开结果", self._open_output)
        self.open_output_button.configure(state="disabled", padx=10, pady=5)
        self.open_output_button.grid(row=0, column=1, padx=(0, 8))
        self.progress = ttk.Progressbar(footer, style="Root.Horizontal.TProgressbar", mode="determinate", length=160)
        self.progress.grid(row=0, column=2, sticky="e", padx=(0, 24))

    def _panel(self, parent, row, column, sticky="nsew", padx=(0, 0), pady=(0, 0)):
        panel = tk.Frame(parent, bg=WHITE, highlightbackground=LINE, highlightthickness=1)
        panel.grid(row=row, column=column, sticky=sticky, padx=padx, pady=pady)
        return panel

    def _analysis_layout(self):
        page = self.analysis_page
        page.grid_columnconfigure(0, weight=0, minsize=416)
        page.grid_columnconfigure(1, weight=1)
        page.grid_rowconfigure(1, weight=1)
        head = tk.Frame(page, bg=BG)
        head.grid(row=0, column=0, columnspan=2, sticky="ew", padx=31, pady=(28, 21))
        label(head, "RHIZOBAG IMAGE ANALYSIS", 8, "bold", TEAL, BG).pack(anchor="w")
        label(head, "根系图像表型分析", 22, "bold", INK, BG).pack(anchor="w", pady=(5, 3))
        label(head, "导入根系图像，自动获得分割、骨架与可追溯的图像描述符。", 10, color=MUTED, bg=BG).pack(anchor="w")

        left_shell = tk.Frame(page, bg=BG)
        left_shell.grid(row=1, column=0, sticky="nsew", padx=(31, 10), pady=(0, 22))
        left_shell.grid_columnconfigure(0, weight=1)
        left_shell.grid_rowconfigure(0, weight=1)
        left_canvas = tk.Canvas(left_shell, bg=BG, highlightthickness=0)
        left_canvas.grid(row=0, column=0, sticky="nsew")
        left_scroll = ttk.Scrollbar(left_shell, orient="vertical", command=left_canvas.yview)
        left_scroll.grid(row=0, column=1, sticky="ns")
        left_canvas.configure(yscrollcommand=left_scroll.set)
        left = tk.Frame(left_canvas, bg=BG)
        left_window = left_canvas.create_window((0, 0), window=left, anchor="nw")
        left.bind("<Configure>", lambda event: left_canvas.configure(scrollregion=left_canvas.bbox("all")))
        left_canvas.bind("<Configure>", lambda event: left_canvas.itemconfigure(left_window, width=event.width))
        left_canvas.bind("<MouseWheel>", lambda event: left_canvas.yview_scroll(-int(event.delta / 120), "units"))
        left.grid_columnconfigure(0, weight=1)
        input_panel = self._panel(left, 0, 0, pady=(0, 14))
        input_panel.grid_columnconfigure(0, weight=1)
        label(input_panel, "01   导入图像", 12, "bold").grid(row=0, column=0, sticky="w", padx=20, pady=(17, 12))
        self.drop_area = tk.Frame(input_panel, bg="#f6fbf8", height=92,
                                  highlightbackground="#bcd9d0", highlightthickness=1, cursor="hand2")
        self.drop_area.grid(row=1, column=0, sticky="ew", padx=20)
        self.drop_area.grid_propagate(False)
        label(self.drop_area, "↥   拖放图片或点击选择", 12, "bold", TEAL, "#f6fbf8").pack(pady=(17, 3))
        label(self.drop_area, "支持 TIFF、PNG、JPG、BMP，可一次导入多张", 8, color=MUTED, bg="#f6fbf8").pack()
        for widget in [self.drop_area] + list(self.drop_area.winfo_children()):
            widget.bind("<Button-1>", lambda event: self._choose_images())
            if DND_FILES:
                try:
                    widget.drop_target_register(DND_FILES)
                    widget.dnd_bind("<<Drop>>", self._drop_images)
                except tk.TclError:
                    pass
        bar = tk.Frame(input_panel, bg=WHITE)
        bar.grid(row=2, column=0, sticky="ew", padx=20, pady=(13, 9))
        button(bar, "选择图片", self._choose_images).pack(side="left")
        button(bar, "导入文件夹", self._choose_folder).pack(side="left", padx=7)
        button(bar, "清空", self._clear_images).pack(side="right")
        list_frame = tk.Frame(input_panel, bg=WHITE)
        list_frame.grid(row=3, column=0, sticky="nsew", padx=20)
        list_frame.grid_columnconfigure(0, weight=1)
        self.input_list = tk.Listbox(list_frame, height=6, relief="flat", bd=0,
                                     bg="#f8faf9", fg=INK, selectbackground=PALE,
                                     font=(FONT, 9), activestyle="none")
        self.input_list.grid(row=0, column=0, sticky="nsew")
        input_scroll = ttk.Scrollbar(list_frame, orient="vertical", command=self.input_list.yview)
        input_scroll.grid(row=0, column=1, sticky="ns")
        self.input_list.configure(yscrollcommand=input_scroll.set)
        self.input_count = label(input_panel, "尚未选择原图", 8, color=MUTED)
        self.input_count.grid(row=4, column=0, sticky="w", padx=20, pady=(9, 13))
        tk.Frame(input_panel, bg=LINE, height=1).grid(row=5, column=0, sticky="ew", padx=20)
        ref_row = tk.Frame(input_panel, bg=WHITE)
        ref_row.grid(row=6, column=0, sticky="ew", padx=20, pady=13)
        label(ref_row, "参考掩膜（可选）", 9, "bold").pack(anchor="w")
        label(ref_row, "同名或带 _mask 后缀；用于 Dice、IoU 等评估。", 8, color=MUTED).pack(anchor="w", pady=(2, 8))
        button(ref_row, "添加参考掩膜", self._choose_references).pack(side="left")
        self.reference_count = label(ref_row, "0 张", 8, color=MUTED)
        self.reference_count.pack(side="left", padx=10)

        settings = self._panel(left, 1, 0)
        settings.grid_columnconfigure(0, weight=1)
        label(settings, "02   分析设置", 12, "bold").grid(row=0, column=0, sticky="w", padx=20, pady=(17, 14))
        model_row = tk.Frame(settings, bg=WHITE)
        model_row.grid(row=1, column=0, sticky="ew", padx=20, pady=(0, 12))
        label(model_row, "研究模型", 9, "bold").pack(side="left")
        choices = [item["id"] for item in catalog()["models"]]
        self.model_choice = ttk.Combobox(model_row, textvariable=self.model_id, values=choices,
                                         state="readonly", width=12)
        self.model_choice.pack(side="right")
        self.model_choice.bind("<<ComboboxSelected>>", self._model_changed)
        threshold_row = tk.Frame(settings, bg=WHITE)
        threshold_row.grid(row=2, column=0, sticky="ew", padx=20)
        label(threshold_row, "分割阈值", 9, "bold").pack(side="left")
        self.threshold_label = label(threshold_row, "0.47", 14, "bold", TEAL)
        self.threshold_label.pack(side="right")
        tk.Scale(settings, variable=self.threshold, from_=0.10, to=0.90, resolution=0.01,
                 orient="horizontal", showvalue=False, command=self._threshold_changed,
                 length=330, troughcolor="#dcebe5", activebackground=TEAL,
                 highlightthickness=0, bd=0, bg=WHITE).grid(row=3, column=0, sticky="ew", padx=20)
        self.threshold_note = label(settings, "当前折验证集阈值 0.47；调整后会写入来源记录。", 8, color=MUTED)
        self.threshold_note.grid(row=4, column=0, sticky="w", padx=20)
        tk.Frame(settings, bg=LINE, height=1).grid(row=5, column=0, sticky="ew", padx=20, pady=14)
        label(settings, "结果保存位置", 9, "bold").grid(row=6, column=0, sticky="w", padx=20)
        output_row = tk.Frame(settings, bg=WHITE)
        output_row.grid(row=7, column=0, sticky="ew", padx=20, pady=(7, 14))
        output_row.grid_columnconfigure(0, weight=1)
        tk.Entry(output_row, textvariable=self.output_folder, relief="solid", bd=1,
                 font=(FONT, 8), fg=INK).grid(row=0, column=0, sticky="ew", ipady=5)
        button(output_row, "浏览", self._choose_output).grid(row=0, column=1, padx=(7, 0))
        ttk.Checkbutton(settings, text="同时保存 float32 概率图", variable=self.save_probability,
                        style="Root.TCheckbutton").grid(row=8, column=0, sticky="w", padx=20)
        label(settings, "结果包含 CSV、掩膜、叠加图、骨架图及来源记录。", 8, color=MUTED).grid(
            row=9, column=0, sticky="w", padx=20, pady=(7, 12))
        action_row = tk.Frame(left_shell, bg=WHITE, highlightbackground=LINE, highlightthickness=1)
        action_row.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(9, 0))
        self.run_button = button(action_row, "开始分析  →", self._start, primary=True)
        self.run_button.pack(side="left", fill="x", expand=True, padx=(13, 0), pady=12)
        self.cancel_button = button(action_row, "取消", self._cancel)
        self.cancel_button.pack(side="left", padx=(7, 13), pady=12)
        self.cancel_button.configure(state="disabled")

        right = tk.Frame(page, bg=BG)
        right.grid(row=1, column=1, sticky="nsew", padx=(10, 31), pady=(0, 22))
        right.grid_columnconfigure(0, weight=1)
        right.grid_rowconfigure(2, weight=1)
        stats = tk.Frame(right, bg=BG)
        stats.grid(row=0, column=0, sticky="ew", pady=(0, 13))
        for col in range(4): stats.grid_columnconfigure(col, weight=1)
        self.stats = {}
        for col, (key, title, unit) in enumerate([
            ("images", "已分析", "张"), ("foreground", "前景比例中位数", "%"),
            ("paths", "有效路径", "张"), ("qc", "需要复核", "张")]):
            card = self._panel(stats, 0, col, padx=(0, 8 if col < 3 else 0))
            label(card, title, 8, color=MUTED).pack(anchor="w", padx=15, pady=(12, 3))
            line = tk.Frame(card, bg=WHITE)
            line.pack(anchor="w", padx=15, pady=(0, 10))
            value = label(line, "—", 19, "bold", INK)
            value.pack(side="left")
            label(line, " " + unit, 8, color=MUTED).pack(side="left", pady=(7, 0))
            self.stats[key] = value
        table_panel = self._panel(right, 1, 0)
        table_panel.grid_columnconfigure(0, weight=1)
        table_panel.grid_rowconfigure(1, weight=1)
        label(table_panel, "03   图像与表型描述符", 12, "bold").grid(row=0, column=0, sticky="w", padx=17, pady=(15, 11))
        columns = ("name", "path", "segments", "junctions", "angle", "qc")
        self.table = ttk.Treeview(table_panel, columns=columns, show="headings", style="Root.Treeview", selectmode="browse", height=5)
        labels = {"name": "图像", "path": "优势路径 / px", "segments": "保留线段",
                  "junctions": "候选连接", "angle": "锐角 / °", "qc": "复核"}
        widths = {"name": 160, "path": 112, "segments": 76, "junctions": 80, "angle": 75, "qc": 65}
        for key in columns:
            self.table.heading(key, text=labels[key])
            self.table.column(key, width=widths[key], minwidth=55, anchor="w" if key == "name" else "center")
        self.table.grid(row=1, column=0, sticky="nsew", padx=(17, 0), pady=(0, 17))
        table_scroll = ttk.Scrollbar(table_panel, orient="vertical", command=self.table.yview)
        table_scroll.grid(row=1, column=1, sticky="ns", padx=(0, 12), pady=(0, 17))
        self.table.configure(yscrollcommand=table_scroll.set)
        self.table.bind("<<TreeviewSelect>>", self._select_result)
        preview_panel = self._panel(right, 2, 0, pady=(13, 0))
        preview_panel.grid_columnconfigure(0, weight=1)
        preview_panel.grid_rowconfigure(2, weight=1)
        preview_head = tk.Frame(preview_panel, bg=WHITE)
        preview_head.grid(row=0, column=0, sticky="ew", padx=17, pady=(14, 8))
        label(preview_head, "04   图像复核", 12, "bold").pack(side="left")
        self.selected_name = label(preview_head, "选择表格中的图像", 8, color=MUTED)
        self.selected_name.pack(side="right")
        tabs = tk.Frame(preview_panel, bg="#eff5f2")
        tabs.grid(row=1, column=0, sticky="w", padx=17, pady=(0, 9))
        self.layer_buttons = {}
        for name in LAYERS:
            tab = tk.Button(tabs, text=name, relief="flat", bd=0, padx=13, pady=5,
                            font=(FONT, 8), cursor="hand2", command=lambda value=name: self._set_layer(value))
            tab.pack(side="left", padx=2, pady=2)
            self.layer_buttons[name] = tab
        self.preview = tk.Label(preview_panel, text="选择一张已分析图像查看结果", bg="#eef4f1", fg="#9aacaa",
                                font=(FONT, 9), compound="center")
        self.preview.grid(row=2, column=0, sticky="nsew", padx=17)
        self.preview.bind("<Configure>", self._preview_resize)
        self.details = label(preview_panel, "长度单位：像素。候选连接区域与保留线段均为图像指标。", 8,
                             color=MUTED, justify="left", wraplength=720)
        self.details.grid(row=3, column=0, sticky="w", padx=17, pady=(8, 13))
        self._set_layer("叠加")

    def _methods_layout(self):
        page = self.methods_page
        page.grid_columnconfigure(0, weight=1)
        page.grid_rowconfigure(1, weight=1)
        head = tk.Frame(page, bg=BG)
        head.grid(row=0, column=0, sticky="ew", padx=31, pady=(28, 20))
        label(head, "METHODS & REPRODUCIBILITY", 8, "bold", TEAL, BG).pack(anchor="w")
        label(head, "方法与研究证据", 22, "bold", INK, BG).pack(anchor="w", pady=(5, 4))
        label(head, "明确模型来源、指标定义与论文交叉验证结果。", 10, color=MUTED, bg=BG).pack(anchor="w")
        panel = self._panel(page, 1, 0, padx=(31, 31), pady=(0, 25))
        panel.grid_columnconfigure(0, weight=1)
        panel.grid_rowconfigure(0, weight=1)
        canvas = tk.Canvas(panel, bg=WHITE, highlightthickness=0)
        canvas.grid(row=0, column=0, sticky="nsew")
        scroll = ttk.Scrollbar(panel, orient="vertical", command=canvas.yview)
        scroll.grid(row=0, column=1, sticky="ns")
        canvas.configure(yscrollcommand=scroll.set)
        body = tk.Frame(canvas, bg=WHITE)
        window = canvas.create_window((0, 0), window=body, anchor="nw")
        body.bind("<Configure>", lambda event: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>", lambda event: canvas.itemconfigure(window, width=event.width))
        sections = [
            ("01   分割模型与推理", "发布包包含论文固定骨架损失 Attention U-Net 的 seed 42 / fold 0–4 五个独立检查点。每个模型使用自身验证集选定的阈值；默认 fold 0。灰度图经 CLAHE（clip 2.0，16×16 网格）和 [0,1] 归一化后，进行 256×256 窗口、128 像素步长预测，采用 σ=64 的高斯权重融合。选择其他折不会自动形成集成模型。"),
            ("02   双通道图像描述符", "通道 A 对原始掩膜骨架化，选取垂直跨度最大的分量，再以八邻接加权最短路径计算优势路径长度。通道 B 用 5×5 椭圆闭合、15 轮末端修剪，计算候选连接区域、至少 10 像素的保留线段和局部锐角均值。无有效路径或角度时记录空值及原因。"),
            ("03   参考掩膜与复核", "若提供同名或带 _mask 后缀的参考掩膜，将计算 Precision、Recall、Dice/F1、IoU、hard clDice 及按参考骨架局部宽度分层的中心线召回率。两种掩膜使用相同的描述符算法，所得误差是图像处理链的误差，不是独立解剖学真值。"),
        ]
        for title, text in sections:
            frame = tk.Frame(body, bg=WHITE)
            frame.pack(fill="x", padx=24, pady=(19, 0))
            label(frame, title, 12, "bold").pack(anchor="w")
            label(frame, text, 9, color=MUTED, wraplength=870, justify="left").pack(anchor="w", pady=(7, 0))
        label(body, "04   论文受控比较", 12, "bold").pack(anchor="w", padx=24, pady=(26, 10))
        cols = ("config", "dice", "iou", "cldice", "fine")
        table = ttk.Treeview(body, columns=cols, show="headings", height=4, style="Root.Treeview")
        for key, title, width in [("config", "配置", 310), ("dice", "Dice", 100),
                                  ("iou", "IoU", 100), ("cldice", "hard clDice", 130),
                                  ("fine", "≤3 px 召回", 130)]:
            table.heading(key, text=title)
            table.column(key, width=width, anchor="w" if key == "config" else "center")
        for values in [
            ("U-Net · 像素损失", "0.5504", "0.3821", "0.6207", "0.5883"),
            ("Attention U-Net · 像素损失", "0.5590", "0.3899", "0.6463", "0.5903"),
            ("Attention U-Net · 固定骨架损失（五折）", "0.5843", "0.4149", "0.6743", "0.6564"),
            ("Attention U-Net · 两阶段候选", "0.5754", "0.4065", "0.6586", "0.6348"),
        ]:
            table.insert("", "end", values=values)
        table.pack(fill="x", padx=24)
        label(body, "表中为 50 张留出图像在三个种子上的汇总均值；该数值不同于任一折模型在新图像上的预期精度。", 8,
              color=MUTED, wraplength=870, justify="left").pack(anchor="w", padx=24, pady=(9, 0))
        label(body, "05   解释边界", 12, "bold").pack(anchor="w", padx=24, pady=(24, 7))
        label(body, "优势路径是像素空间图像指标，不等于总根长或独立验证的解剖学主根长度。保留线段、候选连接区域和局部角度不能直接解释为真实侧根、分枝点和萌发角。软件未使用物理标尺，长度单位仅为像素。跨拍摄条件和跨物种性能仍需外部验证。", 9,
              color=MUTED, wraplength=870, justify="left").pack(anchor="w", padx=24, pady=(0, 28))

    def _shortcuts(self):
        self.bind("<Control-o>", lambda event: self._choose_images())
        self.bind("<Control-r>", lambda event: self._start())

    def _show_page(self, name):
        if name == "analysis":
            self.analysis_page.tkraise()
            self.analysis_nav.configure(bg="#24565b", fg=WHITE, font=(FONT, 11, "bold"))
            self.methods_nav.configure(bg=NAVY, fg="#c1d9d5", font=(FONT, 11))
            self.breadcrumb.configure(text="RootScope  /  图像分析")
        else:
            self.methods_page.tkraise()
            self.methods_nav.configure(bg="#24565b", fg=WHITE, font=(FONT, 11, "bold"))
            self.analysis_nav.configure(bg=NAVY, fg="#c1d9d5", font=(FONT, 11))
            self.breadcrumb.configure(text="RootScope  /  方法与证据")

    def _add_images(self, paths):
        existing = {str(path.resolve()).casefold() for path in self.images}
        for path in paths:
            candidate = Path(path)
            if candidate.is_file() and candidate.suffix.lower() in IMAGE_SUFFIXES and str(candidate.resolve()).casefold() not in existing:
                self.images.append(candidate)
                existing.add(str(candidate.resolve()).casefold())
                self.input_list.insert("end", candidate.name)
        self.input_count.configure(text=f"已选择 {len(self.images)} 张原图")

    def _choose_images(self):
        paths = filedialog.askopenfilenames(title="选择根系图像", filetypes=FILE_TYPES)
        self._add_images(paths)

    def _choose_folder(self):
        folder = filedialog.askdirectory(title="选择根系图像文件夹")
        if folder:
            self._add_images(sorted(path for path in Path(folder).iterdir() if path.suffix.lower() in IMAGE_SUFFIXES))

    def _drop_images(self, event):
        paths = [Path(path) for path in self.tk.splitlist(event.data)]
        files = []
        for path in paths:
            if path.is_dir():
                files.extend(sorted(item for item in path.iterdir() if item.suffix.lower() in IMAGE_SUFFIXES))
            else:
                files.append(path)
        self._add_images(files)

    def _clear_images(self):
        if self.running: return
        self.images.clear()
        self.input_list.delete(0, "end")
        self.input_count.configure(text="尚未选择原图")

    def _choose_references(self):
        paths = filedialog.askopenfilenames(title="选择参考掩膜", filetypes=FILE_TYPES)
        existing = {str(path.resolve()).casefold() for path in self.references}
        for path in paths:
            candidate = Path(path)
            if candidate.is_file() and str(candidate.resolve()).casefold() not in existing:
                self.references.append(candidate)
                existing.add(str(candidate.resolve()).casefold())
        self.reference_count.configure(text=f"{len(self.references)} 张")

    def _choose_output(self):
        folder = filedialog.askdirectory(title="选择结果保存目录")
        if folder:
            self.output_folder.set(folder)

    def _open_output(self):
        if self.last_output is None or not self.last_output.is_dir():
            return
        if sys.platform == "win32":
            os.startfile(self.last_output)
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(self.last_output)])
        else:
            subprocess.Popen(["xdg-open", str(self.last_output)])

    def _threshold_changed(self, value):
        self.threshold_label.configure(text=f"{float(value):.2f}")

    def _model_changed(self, event=None):
        spec = model_spec(self.model_id.get())
        self.threshold.set(float(spec["threshold"]))
        self.threshold_label.configure(text=f"{float(spec['threshold']):.2f}")
        self.threshold_note.configure(text=f"当前折验证集阈值 {float(spec['threshold']):.2f}；调整后会写入来源记录。")
        self.model_sidebar.configure(text=f"固定骨架损失 · Seed 42 · Fold {spec['fold']}")

    def _start(self):
        if self.running: return
        if not self.images:
            messagebox.showinfo("请导入图片", "请先添加至少一张根系图像。")
            return
        output_text = self.output_folder.get().strip()
        if not output_text:
            messagebox.showerror("输出目录", "请选择结果保存目录。")
            return
        folder = Path(output_text).expanduser()
        self.running = True
        self.cancel_event.clear()
        self.model_choice.configure(state="disabled")
        self.run_button.configure(state="disabled")
        self.cancel_button.configure(state="normal")
        self.open_output_button.configure(state="disabled")
        self.progress.configure(value=0, maximum=len(self.images))
        self.status.configure(text="正在加载 ONNX 模型…")
        self.rows.clear()
        self.current_result = None
        self.table.delete(*self.table.get_children())
        for widget in self.stats.values(): widget.configure(text="—")
        inputs = list(self.images)
        references = list(self.references)
        threshold = float(self.threshold.get())
        model_id = self.model_id.get()
        save_probability = self.save_probability.get()
        threading.Thread(target=self._worker, args=(inputs, references, folder, threshold, save_probability, model_id), daemon=True).start()

    def _worker(self, images, references, output, threshold, save_probability, model_id):
        try:
            result = run_batch(images, references, output, threshold, save_probability,
                               self.cancel_event,
                               lambda done, total, text: self.events.put(("progress", done, total, text)),
                               model_id=model_id)
            self.events.put(("complete", result))
        except Exception as error:
            self.events.put(("error", str(error)))

    def _cancel(self):
        if self.running:
            self.cancel_event.set()
            self.status.configure(text="已请求取消；当前图片完成后停止。")
            self.cancel_button.configure(state="disabled")

    def _poll(self):
        try:
            while True:
                event = self.events.get_nowait()
                if event[0] == "progress":
                    _, done, total, text = event
                    self.progress.configure(maximum=max(total, 1), value=done)
                    self.status.configure(text=text)
                elif event[0] == "complete":
                    self._complete(event[1])
                elif event[0] == "error":
                    self.running = False
                    self.model_choice.configure(state="readonly")
                    self.run_button.configure(state="normal")
                    self.cancel_button.configure(state="disabled")
                    self.status.configure(text="分析失败")
                    messagebox.showerror("分析失败", event[1])
                    if self.pending_close: self.destroy()
        except queue.Empty:
            pass
        if self.winfo_exists(): self.after(100, self._poll)

    def _complete(self, result):
        self.running = False
        self.model_choice.configure(state="readonly")
        self.run_button.configure(state="normal")
        self.cancel_button.configure(state="disabled")
        self.last_output = Path(result["folder"])
        self.last_archive = Path(result["archive"]) if result["archive"] else None
        self.open_output_button.configure(state="normal")
        rows = result["results"]
        errors = result["errors"]
        fractions = sorted(item["foreground_fraction"] for item in rows)
        self.stats["images"].configure(text=str(len(rows)))
        self.stats["foreground"].configure(text=f"{100 * fractions[len(fractions)//2]:.2f}" if fractions else "—")
        self.stats["paths"].configure(text=str(sum(item["descriptors"]["path_status"] == "ok" for item in rows)))
        self.stats["qc"].configure(text=str(sum(bool(item["qc_flags"]) for item in rows)))
        for item in rows:
            descriptors = item["descriptors"]
            def shown(value):
                return "—" if value is None else f"{value:.1f}"
            iid = item["image_id"]
            self.table.insert("", "end", iid=iid,
                              values=(item["image_name"], shown(descriptors["dominant_path_length"]),
                                      descriptors["retained_segment_count"], descriptors["junction_region_count"],
                                      shown(descriptors["mean_local_acute_angle_deg"]),
                                      "需复核" if item["qc_flags"] else "正常"))
            self.rows[iid] = item
        if rows:
            self.table.selection_set(rows[0]["image_id"])
            self.table.focus(rows[0]["image_id"])
            self._select_result()
        suffix = f"；{len(errors)} 张失败，详见 errors.csv" if errors else ""
        if result["report"]["status"] == "cancelled": suffix += "；任务已取消"
        self.status.configure(text=f"完成：{len(rows)} 张图像{suffix}。结果：{self.last_output}")
        if not rows and errors:
            messagebox.showwarning("没有成功结果", f"全部 {len(errors)} 张图像分析失败。请查看结果目录中的 errors.csv。")
        if self.pending_close: self.destroy()

    def _select_result(self, event=None):
        selection = self.table.selection()
        if not selection: return
        self.current_result = self.rows.get(selection[0])
        if self.current_result is None: return
        item = self.current_result
        self.selected_name.configure(text=item["image_name"])
        descriptors = item["descriptors"]
        qc = ", ".join(item["qc_flags"]) if item["qc_flags"] else "无"
        detail = f"尺寸 {item['width']}×{item['height']} px   ·   路径状态 {descriptors['path_status']}   ·   有效角度 {descriptors['valid_angle_n']} 个   ·   质控 {qc}"
        if "evaluation" in item:
            evaluation = item["evaluation"]
            def metric(value):
                return "—" if value is None else f"{value:.3f}"
            detail += (f"\n参考掩膜：Dice {metric(evaluation['dice'])}   ·   "
                       f"IoU {metric(evaluation['iou'])}   ·   hard clDice {metric(evaluation['hard_cldice'])}")
        self.details.configure(text=detail)
        self._refresh_preview()

    def _set_layer(self, name):
        self.layer.set(name)
        for key, widget in self.layer_buttons.items():
            widget.configure(bg=WHITE if key == name else "#eff5f2",
                             fg=TEAL if key == name else MUTED)
        self._refresh_preview()

    def _preview_resize(self, event):
        if self.preview_after is not None:
            self.after_cancel(self.preview_after)
        self.preview_after = self.after(120, self._refresh_preview)

    def _refresh_preview(self):
        self.preview_after = None
        if self.current_result is None or self.last_output is None:
            return
        item = self.current_result
        layer = LAYERS[self.layer.get()]
        path = Path(item["_source_path"]) if layer is None else self.last_output / "images" / item["image_id"] / layer
        if not path.exists():
            self.preview.configure(text="图像文件不可用", image="")
            return
        try:
            with Image.open(path) as image:
                picture = image.convert("RGB")
                width = max(100, self.preview.winfo_width() - 18)
                height = max(100, self.preview.winfo_height() - 18)
                picture.thumbnail((width, height), Image.Resampling.LANCZOS)
                self.preview_image = ImageTk.PhotoImage(picture, master=self)
            self.preview.configure(image=self.preview_image, text="")
        except Exception as error:
            self.preview.configure(text=f"预览失败：{error}", image="")

    def _close(self):
        if self.running:
            if messagebox.askyesno("正在分析", "当前任务正在运行。处理完当前图片后退出吗？"):
                self.pending_close = True
                self._cancel()
            return
        self.destroy()


def main():
    if len(sys.argv) >= 2 and sys.argv[1] == "--batch":
        from cli import main as cli_main

        sys.exit(cli_main(sys.argv[2:]))
    if len(sys.argv) >= 2 and sys.argv[1] == "--self-test":
        import numpy as np

        from rootscope.engine import InferenceEngine
        from rootscope.models import catalog, model_path

        checked = []
        for spec in catalog()["models"]:
            engine = InferenceEngine(model_path(spec["id"]), spec["sha256"])
            probability = engine.predict(np.zeros((256, 256), dtype=np.uint8))
            if probability.shape != (256, 256) or not np.isfinite(probability).all():
                raise RuntimeError(f"Self-test failed: invalid output for {spec['id']}")
            checked.append({"id": spec["id"], "sha256": spec["sha256"], "provider": engine.device})
        report = {"status": "ok", "version": __version__, "models": checked,
                  "shape": [256, 256]}
        if len(sys.argv) >= 3:
            Path(sys.argv[2]).write_text(json.dumps(report, indent=2), encoding="utf-8")
        else:
            print(json.dumps(report))
        return
    if sys.platform == "win32":
        try:
            import ctypes
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except (AttributeError, OSError):
            pass
    RootScopeApp().mainloop()


if __name__ == "__main__":
    main()
