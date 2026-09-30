"""RootScope native desktop application. No browser or local web server is used."""
from __future__ import annotations
from rootscope.i18n import tr, set_language, get_language, load_language, save_language
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
BG = '#f4f8f6'
WHITE = '#ffffff'
INK = '#193544'
MUTED = '#698085'
NAVY = '#113841'
TEAL = '#147e78'
PALE = '#e7f4ef'
LINE = '#dfe9e5'
AMBER = '#b76d3b'
FONT = 'Microsoft YaHei UI' if sys.platform == 'win32' else 'DejaVu Sans'
FILE_TYPES = [(tr('text_87a32e9b0a9a'), '*.tif *.tiff *.png *.jpg *.jpeg *.bmp'), (tr('text_46ba809193bd'), '*.*')]
LAYERS = {'source': None, 'overlay': 'overlay.png', 'mask': 'prediction_mask.png', 'skeleton': 'skeleton.png'}
LAYER_LABELS = {'source': tr('text_cd0f3ad8ddd8'), 'overlay': tr('text_072afa8c3d7b'), 'mask': tr('text_98cb32da93e9'), 'skeleton': tr('text_6314d4d17442')}

def label(parent, text, size=10, weight='normal', color=INK, bg=WHITE, **kwargs):
    if len(text) > 45 and 'wraplength' not in kwargs:
        kwargs.update(wraplength=330, justify='left')
    return tk.Label(parent, text=text, font=(FONT, size, weight), fg=color, bg=bg, **kwargs)

def button(parent, text, command, primary=False, width=None):
    return tk.Button(parent, text=text, command=command, relief='flat', bd=0, font=(FONT, 10, 'bold'), padx=9, pady=9, width=width, cursor='hand2', activeforeground=WHITE if primary else TEAL, activebackground='#106b66' if primary else '#eff7f3', fg=WHITE if primary else TEAL, bg=TEAL if primary else PALE)

class RootScopeApp(Window):

    def __init__(self, language=None):
        super().__init__()
        set_language(language or load_language())
        self.language = get_language()
        self.last_result = None
        self.current_page = 'analysis'
        self._update_language_resources()
        self.title(tr('text_ade46984c67e'))
        self.geometry('1460x960')
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
        self.output_folder = tk.StringVar(value=str(Path.home() / 'Documents' / 'RootScope Results'))
        self.layer = tk.StringVar(value='overlay')
        self._style()
        self._layout()
        self._shortcuts()
        self.after(100, self._poll)
        self.protocol('WM_DELETE_WINDOW', self._close)

    def _update_language_resources(self):
        global FONT, FILE_TYPES, LAYER_LABELS
        from tkinter import font
        available = {name.casefold():name for name in font.families(self)}
        requested = 'Segoe UI' if self.language == 'en' else 'Microsoft YaHei UI'
        alternatives = [requested, 'Arial' if self.language == 'en' else 'Microsoft YaHei']
        FONT = next((available[name.casefold()] for name in alternatives if name.casefold() in available), font.nametofont('TkDefaultFont').actual('family'))
        FILE_TYPES = [(tr('text_87a32e9b0a9a'), '*.tif *.tiff *.png *.jpg *.jpeg *.bmp'), (tr('text_46ba809193bd'), '*.*')]
        LAYER_LABELS = {'source': tr('text_cd0f3ad8ddd8'), 'overlay': tr('text_072afa8c3d7b'), 'mask': tr('text_98cb32da93e9'), 'skeleton': tr('text_6314d4d17442')}

    def _change_language(self, event=None):
        if self.running:
            return
        language = 'en' if self.language_choice.get() == 'English' else 'zh'
        if language == self.language:
            return
        selected = self.table.selection()
        layer = self.layer.get()
        page = self.current_page
        if self.preview_after is not None:
            self.after_cancel(self.preview_after)
            self.preview_after = None
        self.language = language
        set_language(language)
        save_language(language)
        self._update_language_resources()
        for widget in self.winfo_children():
            widget.destroy()
        self.title(tr('text_ade46984c67e'))
        self._style()
        self._layout()
        for path in self.images:
            self.input_list.insert('end', path.name)
        self.input_count.configure(text=tr('text_9f934685f984', p0=str(len(self.images))))
        self.reference_count.configure(text=tr('text_30a97706c37b', p0=str(len(self.references))))
        threshold = self.threshold.get()
        self._model_changed()
        self.threshold.set(threshold)
        self._threshold_changed(threshold)
        if self.last_result:
            self._complete(self.last_result)
            if selected and selected[0] in self.rows:
                self.table.selection_set(selected[0])
                self._select_result()
        self._set_layer(layer)
        self._show_page(page)

    def _style(self):
        style = ttk.Style(self)
        style.theme_use('clam')
        style.configure('Root.Treeview', font=(FONT, 10), rowheight=31, background=WHITE, fieldbackground=WHITE, foreground=INK, borderwidth=0)
        style.configure('Root.Treeview.Heading', font=(FONT, 9, 'bold'), background='#eef4f1', foreground='#567176', padding=8, relief='flat')
        style.map('Root.Treeview', background=[('selected', PALE)], foreground=[('selected', INK)])
        style.configure('Root.Horizontal.TProgressbar', troughcolor='#e0eee8', background=TEAL, bordercolor='#e0eee8', lightcolor=TEAL, darkcolor=TEAL)
        style.configure('Root.TCheckbutton', background=WHITE, foreground=INK, font=(FONT, 9))

    def _layout(self):
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        sidebar = tk.Frame(self, bg=NAVY, width=226)
        sidebar.grid(row=0, column=0, sticky='ns')
        sidebar.grid_propagate(False)
        self.language_choice = ttk.Combobox(sidebar, values=['English', '中文'], state='readonly', width=17)
        self.language_choice.set('English' if self.language == 'en' else '中文')
        self.language_choice.pack(side='bottom', padx=20, pady=12)
        self.language_choice.bind('<<ComboboxSelected>>', self._change_language)
        label(sidebar, '◈  RootScope', 18, 'bold', WHITE, NAVY).pack(anchor='w', padx=25, pady=(31, 2))
        label(sidebar, 'RESEARCH DESKTOP', 8, 'bold', '#9ebcb8', NAVY).pack(anchor='w', padx=54)
        label(sidebar, tr('text_1bf3defde730'), 9, 'bold', '#8aadaa', NAVY).pack(anchor='w', padx=24, pady=(58, 13))
        self.analysis_nav = tk.Button(sidebar, text=tr('text_c59591205d33'), command=lambda: self._show_page('analysis'), anchor='w', padx=20, pady=12, bd=0, relief='flat', font=(FONT, 11, 'bold'), fg=WHITE, bg='#24565b', cursor='hand2')
        self.analysis_nav.pack(fill='x', padx=13, pady=3)
        self.methods_nav = tk.Button(sidebar, text=tr('text_376085a4d383'), command=lambda: self._show_page('methods'), anchor='w', padx=20, pady=12, bd=0, relief='flat', font=(FONT, 11), fg='#c1d9d5', bg=NAVY, cursor='hand2')
        self.methods_nav.pack(fill='x', padx=13, pady=3)
        tk.Frame(sidebar, bg='#2e5c5e', height=1).pack(fill='x', padx=25, pady=30)
        label(sidebar, tr('text_5af4c063583e'), 9, 'bold', '#8aadaa', NAVY).pack(anchor='w', padx=24)
        label(sidebar, '●  Attention U-Net', 10, 'bold', '#e2f3ed', NAVY).pack(anchor='w', padx=24, pady=(16, 3))
        self.model_sidebar = label(sidebar, tr('text_2e1d307d321f'), 8, color='#a9c8c3', bg=NAVY)
        self.model_sidebar.pack(anchor='w', padx=41)
        label(sidebar, tr('text_75a14730934c'), 8, color='#a6c6c0', bg=NAVY).pack(side='bottom', anchor='w', padx=24, pady=28)
        outer = tk.Frame(self, bg=BG)
        outer.grid(row=0, column=1, sticky='nsew')
        outer.grid_rowconfigure(1, weight=1)
        outer.grid_columnconfigure(0, weight=1)
        topbar = tk.Frame(outer, bg=WHITE, height=59, highlightbackground=LINE, highlightthickness=1)
        topbar.grid(row=0, column=0, sticky='ew')
        topbar.grid_propagate(False)
        self.breadcrumb = label(topbar, tr('text_ea69ab24bbaa'), 9, color=MUTED, bg=WHITE)
        self.breadcrumb.pack(side='left', padx=32)
        label(topbar, tr('text_dd02c56cd370'), 9, color=TEAL, bg=WHITE).pack(side='right', padx=32)
        self.pages = tk.Frame(outer, bg=BG)
        self.pages.grid(row=1, column=0, sticky='nsew')
        self.pages.grid_rowconfigure(0, weight=1)
        self.pages.grid_columnconfigure(0, weight=1)
        self.analysis_page = tk.Frame(self.pages, bg=BG)
        self.methods_page = tk.Frame(self.pages, bg=BG)
        self.analysis_page.grid(row=0, column=0, sticky='nsew')
        self.methods_page.grid(row=0, column=0, sticky='nsew')
        self._analysis_layout()
        self._methods_layout()
        self._show_page('analysis')
        footer = tk.Frame(outer, bg=WHITE, height=54, highlightbackground=LINE, highlightthickness=1)
        footer.grid(row=2, column=0, sticky='ew')
        footer.grid_columnconfigure(0, weight=1)
        self.status = label(footer, tr('text_26bf8c801559'), 9, color=MUTED, bg=WHITE)
        self.status.grid(row=0, column=0, sticky='w', padx=25, pady=15)
        self.open_output_button = button(footer, tr('text_189046f73761'), self._open_output)
        self.open_output_button.configure(state='disabled', padx=10, pady=5)
        self.open_output_button.grid(row=0, column=1, padx=(0, 8))
        self.progress = ttk.Progressbar(footer, style='Root.Horizontal.TProgressbar', mode='determinate', length=160)
        self.progress.grid(row=0, column=2, sticky='e', padx=(0, 24))

    def _panel(self, parent, row, column, sticky='nsew', padx=(0, 0), pady=(0, 0)):
        panel = tk.Frame(parent, bg=WHITE, highlightbackground=LINE, highlightthickness=1)
        panel.grid(row=row, column=column, sticky=sticky, padx=padx, pady=pady)
        return panel

    def _analysis_layout(self):
        page = self.analysis_page
        page.grid_columnconfigure(0, weight=0, minsize=440)
        page.grid_columnconfigure(1, weight=1)
        page.grid_rowconfigure(1, weight=1)
        head = tk.Frame(page, bg=BG)
        head.grid(row=0, column=0, columnspan=2, sticky='ew', padx=31, pady=(28, 21))
        label(head, 'RHIZOBAG IMAGE ANALYSIS', 8, 'bold', TEAL, BG).pack(anchor='w')
        label(head, tr('text_049d50a4d293'), 22, 'bold', INK, BG).pack(anchor='w', pady=(5, 3))
        label(head, tr('text_70b87918773e'), 10, color=MUTED, bg=BG).pack(anchor='w')
        left_shell = tk.Frame(page, bg=BG)
        left_shell.grid(row=1, column=0, sticky='nsew', padx=(31, 10), pady=(0, 22))
        left_shell.grid_columnconfigure(0, weight=1)
        left_shell.grid_rowconfigure(0, weight=1)
        left_canvas = tk.Canvas(left_shell, bg=BG, highlightthickness=0)
        self.settings_canvas = left_canvas
        left_canvas.grid(row=0, column=0, sticky='nsew')
        left_scroll = ttk.Scrollbar(left_shell, orient='vertical', command=left_canvas.yview)
        left_scroll.grid(row=0, column=1, sticky='ns')
        left_canvas.configure(yscrollcommand=left_scroll.set)
        left = tk.Frame(left_canvas, bg=BG)
        left_window = left_canvas.create_window((0, 0), window=left, anchor='nw')
        left.bind('<Configure>', lambda event: left_canvas.configure(scrollregion=left_canvas.bbox('all')))
        left_canvas.bind('<Configure>', lambda event: left_canvas.itemconfigure(left_window, width=event.width))
        left_canvas.bind('<MouseWheel>', lambda event: left_canvas.yview_scroll(-int(event.delta / 120), 'units'))
        left.grid_columnconfigure(0, weight=1)
        input_panel = self._panel(left, 0, 0, pady=(0, 14))
        input_panel.grid_columnconfigure(0, weight=1)
        label(input_panel, tr('text_85abcd952e82'), 12, 'bold').grid(row=0, column=0, sticky='w', padx=20, pady=(17, 12))
        self.drop_area = tk.Frame(input_panel, bg='#f6fbf8', height=92, highlightbackground='#bcd9d0', highlightthickness=1, cursor='hand2')
        self.drop_area.grid(row=1, column=0, sticky='ew', padx=20)
        self.drop_area.grid_propagate(False)
        label(self.drop_area, tr('text_022f2f166b65'), 12, 'bold', TEAL, '#f6fbf8').pack(pady=(17, 3))
        label(self.drop_area, tr('text_28d6c941a1f8'), 8, color=MUTED, bg='#f6fbf8').pack()
        for widget in [self.drop_area] + list(self.drop_area.winfo_children()):
            widget.bind('<Button-1>', lambda event: self._choose_images())
            if DND_FILES:
                try:
                    widget.drop_target_register(DND_FILES)
                    widget.dnd_bind('<<Drop>>', self._drop_images)
                except tk.TclError:
                    pass
        bar = tk.Frame(input_panel, bg=WHITE)
        bar.grid(row=2, column=0, sticky='ew', padx=20, pady=(13, 9))
        button(bar, tr('text_18104edf8911'), self._choose_images).pack(side='left')
        button(bar, tr('text_4d2d93e528f9'), self._choose_folder).pack(side='left', padx=7)
        button(bar, tr('text_84fcd70d4280'), self._clear_images).pack(side='right')
        list_frame = tk.Frame(input_panel, bg=WHITE)
        list_frame.grid(row=3, column=0, sticky='nsew', padx=20)
        list_frame.grid_columnconfigure(0, weight=1)
        self.input_list = tk.Listbox(list_frame, height=6, relief='flat', bd=0, bg='#f8faf9', fg=INK, selectbackground=PALE, font=(FONT, 9), activestyle='none')
        self.input_list.grid(row=0, column=0, sticky='nsew')
        input_scroll = ttk.Scrollbar(list_frame, orient='vertical', command=self.input_list.yview)
        input_scroll.grid(row=0, column=1, sticky='ns')
        self.input_list.configure(yscrollcommand=input_scroll.set)
        self.input_count = label(input_panel, tr('text_85f2adcfb9db'), 8, color=MUTED)
        self.input_count.grid(row=4, column=0, sticky='w', padx=20, pady=(9, 13))
        tk.Frame(input_panel, bg=LINE, height=1).grid(row=5, column=0, sticky='ew', padx=20)
        ref_row = tk.Frame(input_panel, bg=WHITE)
        ref_row.grid(row=6, column=0, sticky='ew', padx=20, pady=13)
        label(ref_row, tr('text_95065ef282d8'), 9, 'bold').pack(anchor='w')
        label(ref_row, tr('text_4374cdaa6a9b'), 8, color=MUTED).pack(anchor='w', pady=(2, 8))
        button(ref_row, tr('text_b589cfb36b8c'), self._choose_references).pack(side='left')
        self.reference_count = label(ref_row, tr('text_10db60b2bf2a'), 8, color=MUTED)
        self.reference_count.pack(side='left', padx=10)
        settings = self._panel(left, 1, 0)
        settings.grid_columnconfigure(0, weight=1)
        label(settings, tr('text_9a81b8b31c39'), 12, 'bold').grid(row=0, column=0, sticky='w', padx=20, pady=(17, 14))
        model_row = tk.Frame(settings, bg=WHITE)
        model_row.grid(row=1, column=0, sticky='ew', padx=20, pady=(0, 12))
        label(model_row, tr('text_5af4c063583e'), 9, 'bold').pack(side='left')
        choices = [item['id'] for item in catalog()['models']]
        self.model_choice = ttk.Combobox(model_row, textvariable=self.model_id, values=choices, state='readonly', width=12)
        self.model_choice.pack(side='right')
        self.model_choice.bind('<<ComboboxSelected>>', self._model_changed)
        threshold_row = tk.Frame(settings, bg=WHITE)
        threshold_row.grid(row=2, column=0, sticky='ew', padx=20)
        label(threshold_row, tr('text_bd4262205c03'), 9, 'bold').pack(side='left')
        self.threshold_label = label(threshold_row, '0.47', 14, 'bold', TEAL)
        self.threshold_label.pack(side='right')
        tk.Scale(settings, variable=self.threshold, from_=0.1, to=0.9, resolution=0.01, orient='horizontal', showvalue=False, command=self._threshold_changed, length=330, troughcolor='#dcebe5', activebackground=TEAL, highlightthickness=0, bd=0, bg=WHITE).grid(row=3, column=0, sticky='ew', padx=20)
        self.threshold_note = label(settings, tr('text_985837893288'), 8, color=MUTED)
        self.threshold_note.grid(row=4, column=0, sticky='w', padx=20)
        tk.Frame(settings, bg=LINE, height=1).grid(row=5, column=0, sticky='ew', padx=20, pady=14)
        label(settings, tr('text_cd9224e3fd63'), 9, 'bold').grid(row=6, column=0, sticky='w', padx=20)
        output_row = tk.Frame(settings, bg=WHITE)
        output_row.grid(row=7, column=0, sticky='ew', padx=20, pady=(7, 14))
        output_row.grid_columnconfigure(0, weight=1)
        tk.Entry(output_row, textvariable=self.output_folder, relief='solid', bd=1, font=(FONT, 8), fg=INK).grid(row=0, column=0, sticky='ew', ipady=5)
        button(output_row, tr('text_ef23ea67b002'), self._choose_output).grid(row=0, column=1, padx=(7, 0))
        ttk.Checkbutton(settings, text=tr('text_9af5a38132a1'), variable=self.save_probability, style='Root.TCheckbutton').grid(row=8, column=0, sticky='w', padx=20)
        label(settings, tr('text_5e0477856430'), 8, color=MUTED).grid(row=9, column=0, sticky='w', padx=20, pady=(7, 12))
        action_row = tk.Frame(left_shell, bg=WHITE, highlightbackground=LINE, highlightthickness=1)
        action_row.grid(row=1, column=0, columnspan=2, sticky='ew', pady=(9, 0))
        self.run_button = button(action_row, tr('text_67c45ac47d3b'), self._start, primary=True)
        self.run_button.pack(side='left', fill='x', expand=True, padx=(13, 0), pady=12)
        self.cancel_button = button(action_row, tr('text_4d0b4688c787'), self._cancel)
        self.cancel_button.pack(side='left', padx=(7, 13), pady=12)
        self.cancel_button.configure(state='disabled')
        right_shell = tk.Frame(page, bg=BG)
        right_shell.grid(row=1, column=1, sticky='nsew', padx=(10, 31), pady=(0, 22))
        right_shell.grid_rowconfigure(0, weight=1)
        right_shell.grid_columnconfigure(0, weight=1)
        right_canvas = tk.Canvas(right_shell, bg=BG, highlightthickness=0)
        right_canvas.grid(row=0, column=0, sticky='nsew')
        right_scroll = ttk.Scrollbar(right_shell, orient='vertical', command=right_canvas.yview)
        right_scroll.grid(row=0, column=1, sticky='ns')
        right_canvas.configure(yscrollcommand=right_scroll.set)
        right = tk.Frame(right_canvas, bg=BG)
        right_window = right_canvas.create_window((0, 0), window=right, anchor='nw')
        right.bind('<Configure>', lambda event: right_canvas.configure(scrollregion=right_canvas.bbox('all')))
        right_canvas.bind('<Configure>', lambda event: right_canvas.itemconfigure(right_window, width=event.width, height=max(event.height, right.winfo_reqheight())))
        right_canvas.bind('<MouseWheel>', lambda event: right_canvas.yview_scroll(-int(event.delta / 120), 'units'))
        right.grid_columnconfigure(0, weight=1)
        right.grid_rowconfigure(2, weight=1)
        stats = tk.Frame(right, bg=BG)
        stats.grid(row=0, column=0, sticky='ew', pady=(0, 13))
        for col in range(4):
            stats.grid_columnconfigure(col, weight=1)
        self.stats = {}
        for col, (key, title, unit) in enumerate([('images', tr('text_38d69c8d2007'), tr('text_9273cc90ea18')), ('foreground', tr('text_3071b58c6332'), '%'), ('paths', tr('text_6e90ea04152c'), tr('text_9273cc90ea18')), ('qc', tr('text_7252b1f25cff'), tr('text_9273cc90ea18'))]):
            card = self._panel(stats, 0, col, padx=(0, 8 if col < 3 else 0))
            card_title = label(card, title, 8, color=MUTED, wraplength=100, justify='left')
            card_title.pack(anchor='w', padx=10, pady=(12, 3))
            card.bind('<Configure>', lambda event, widget=card_title: widget.configure(wraplength=max(35,event.width-20)))
            line = tk.Frame(card, bg=WHITE)
            line.pack(anchor='w', padx=15, pady=(0, 10))
            value = label(line, '—', 19, 'bold', INK)
            value.pack(side='left')
            label(line, ' ' + unit, 8, color=MUTED).pack(side='left', pady=(7, 0))
            self.stats[key] = value
        table_panel = self._panel(right, 1, 0)
        table_panel.grid_columnconfigure(0, weight=1)
        table_panel.grid_rowconfigure(1, weight=1)
        label(table_panel, tr('text_a9ee0f8b9c9b'), 12, 'bold').grid(row=0, column=0, sticky='w', padx=17, pady=(15, 11))
        columns = ('name', 'path', 'segments', 'junctions', 'angle', 'qc')
        self.table = ttk.Treeview(table_panel, columns=columns, show='headings', style='Root.Treeview', selectmode='browse', height=5)
        labels = {'name': tr('text_0a0ce84ddefc'), 'path': tr('text_a6ec3bcf9764'), 'segments': tr('text_2655cdf66ea8'), 'junctions': tr('text_11ecc3977538'), 'angle': tr('text_ea4d5246424e'), 'qc': tr('text_b38c92c32df6')}
        widths = {'name': 160, 'path': 160, 'segments': 145, 'junctions': 155, 'angle': 160, 'qc': 145}
        for key in columns:
            self.table.heading(key, text=labels[key])
            self.table.column(key, width=widths[key], minwidth=widths[key], stretch=False, anchor='w' if key == 'name' else 'center')
        self.table.grid(row=1, column=0, sticky='nsew', padx=(17, 0), pady=(0, 17))
        table_scroll = ttk.Scrollbar(table_panel, orient='vertical', command=self.table.yview)
        table_scroll.grid(row=1, column=1, sticky='ns', padx=(0, 12), pady=(0, 17))
        self.table.configure(yscrollcommand=table_scroll.set)
        table_horizontal = ttk.Scrollbar(table_panel, orient='horizontal', command=self.table.xview)
        table_horizontal.grid(row=2, column=0, sticky='ew', padx=17, pady=(0, 8))
        self.table.configure(xscrollcommand=table_horizontal.set)
        self.table.bind('<<TreeviewSelect>>', self._select_result)
        preview_panel = self._panel(right, 2, 0, pady=(13, 0))
        preview_panel.grid_columnconfigure(0, weight=1)
        preview_panel.grid_rowconfigure(2, weight=1)
        preview_head = tk.Frame(preview_panel, bg=WHITE)
        preview_head.grid(row=0, column=0, sticky='ew', padx=17, pady=(14, 8))
        label(preview_head, tr('text_c54f3ef4812c'), 12, 'bold').pack(side='left')
        self.selected_name = label(preview_head, tr('text_c3d73772b600'), 8, color=MUTED)
        self.selected_name.pack(side='right')
        tabs = tk.Frame(preview_panel, bg='#eff5f2')
        tabs.grid(row=1, column=0, sticky='w', padx=17, pady=(0, 9))
        self.layer_buttons = {}
        for name in LAYERS:
            tab = tk.Button(tabs, text=LAYER_LABELS[name], relief='flat', bd=0, padx=13, pady=5, font=(FONT, 8), cursor='hand2', command=lambda value=name: self._set_layer(value))
            tab.pack(side='left', padx=2, pady=2)
            self.layer_buttons[name] = tab
        self.preview = tk.Label(preview_panel, text=tr('text_0928aae75643'), bg='#eef4f1', fg='#9aacaa', font=(FONT, 9), compound='center')
        self.preview.grid(row=2, column=0, sticky='nsew', padx=17)
        self.preview.bind('<Configure>', self._preview_resize)
        self.details = label(preview_panel, tr('text_45064ca8c50e'), 8, color=MUTED, justify='left', wraplength=720)
        self.details.grid(row=3, column=0, sticky='w', padx=17, pady=(8, 13))
        self._set_layer('overlay')

    def _methods_layout(self):
        page = self.methods_page
        page.grid_columnconfigure(0, weight=1)
        page.grid_rowconfigure(1, weight=1)
        head = tk.Frame(page, bg=BG)
        head.grid(row=0, column=0, sticky='ew', padx=31, pady=(28, 20))
        label(head, 'METHODS & REPRODUCIBILITY', 8, 'bold', TEAL, BG).pack(anchor='w')
        label(head, tr('text_ca7bef237655'), 22, 'bold', INK, BG).pack(anchor='w', pady=(5, 4))
        label(head, tr('text_d11150b45b0f'), 10, color=MUTED, bg=BG).pack(anchor='w')
        panel = self._panel(page, 1, 0, padx=(31, 31), pady=(0, 25))
        panel.grid_columnconfigure(0, weight=1)
        panel.grid_rowconfigure(0, weight=1)
        canvas = tk.Canvas(panel, bg=WHITE, highlightthickness=0)
        canvas.grid(row=0, column=0, sticky='nsew')
        scroll = ttk.Scrollbar(panel, orient='vertical', command=canvas.yview)
        scroll.grid(row=0, column=1, sticky='ns')
        canvas.configure(yscrollcommand=scroll.set)
        body = tk.Frame(canvas, bg=WHITE)
        window = canvas.create_window((0, 0), window=body, anchor='nw')
        body.bind('<Configure>', lambda event: canvas.configure(scrollregion=canvas.bbox('all')))
        canvas.bind('<Configure>', lambda event: canvas.itemconfigure(window, width=event.width))
        sections = [(tr('text_435181f7240d'), tr('text_12876f9dbca9')), (tr('text_fb982e528d64'), tr('text_ba8a9d129592')), (tr('text_546eb27619da'), tr('text_127ecaed4a0c'))]
        for title, text in sections:
            frame = tk.Frame(body, bg=WHITE)
            frame.pack(fill='x', padx=24, pady=(19, 0))
            label(frame, title, 12, 'bold').pack(anchor='w')
            description = label(frame, text, 9, color=MUTED, wraplength=870, justify='left')
            description.pack(anchor='w', pady=(7, 0))
            frame.bind('<Configure>', lambda event, widget=description: widget.configure(wraplength=max(120,event.width-4)))
        label(body, tr('text_ae1b7597035f'), 12, 'bold').pack(anchor='w', padx=24, pady=(26, 10))
        cols = ('config', 'dice', 'iou', 'cldice', 'fine')
        table = ttk.Treeview(body, columns=cols, show='headings', height=4, style='Root.Treeview')
        for key, title, width in [('config', tr('text_d7d7ce790b9a'), 310), ('dice', 'Dice', 100), ('iou', 'IoU', 100), ('cldice', 'hard clDice', 130), ('fine', tr('text_87a6d54d5aaf'), 130)]:
            table.heading(key, text=title)
            table.column(key, width=width, anchor='w' if key == 'config' else 'center')
        for values in [(tr('text_4b604be4f560'), '0.5504', '0.3821', '0.6207', '0.5883'), (tr('text_fdd5429cb311'), '0.5590', '0.3899', '0.6463', '0.5903'), (tr('text_de47abdc7362'), '0.5843', '0.4149', '0.6743', '0.6564'), (tr('text_9190cf526558'), '0.5754', '0.4065', '0.6586', '0.6348')]:
            table.insert('', 'end', values=values)
        table.pack(fill='x', padx=24)
        horizontal = ttk.Scrollbar(body, orient='horizontal', command=table.xview)
        horizontal.pack(fill='x', padx=24)
        table.configure(xscrollcommand=horizontal.set)
        study_note = label(body, tr('text_ba05c6039832'), 8, color=MUTED, wraplength=870, justify='left')
        study_note.pack(anchor='w', padx=24, pady=(9, 0))
        label(body, tr('text_0a6bad6aac69'), 12, 'bold').pack(anchor='w', padx=24, pady=(24, 7))
        interpretation = label(body, tr('text_07ff08ebf446'), 9, color=MUTED, wraplength=870, justify='left')
        interpretation.pack(anchor='w', padx=24, pady=(0, 28))
        body.bind('<Configure>', lambda event: (canvas.configure(scrollregion=canvas.bbox('all')), study_note.configure(wraplength=max(120,event.width-48)), interpretation.configure(wraplength=max(120,event.width-48))))

    def _shortcuts(self):
        self.bind('<Control-o>', lambda event: self._choose_images())
        self.bind('<Control-r>', lambda event: self._start())

    def _show_page(self, name):
        self.current_page = name
        if name == 'analysis':
            self.analysis_page.tkraise()
            self.analysis_nav.configure(bg='#24565b', fg=WHITE, font=(FONT, 11, 'bold'))
            self.methods_nav.configure(bg=NAVY, fg='#c1d9d5', font=(FONT, 11))
            self.breadcrumb.configure(text=tr('text_ea69ab24bbaa'))
        else:
            self.methods_page.tkraise()
            self.methods_nav.configure(bg='#24565b', fg=WHITE, font=(FONT, 11, 'bold'))
            self.analysis_nav.configure(bg=NAVY, fg='#c1d9d5', font=(FONT, 11))
            self.breadcrumb.configure(text=tr('text_f27f162f13bf'))

    def _add_images(self, paths):
        existing = {str(path.resolve()).casefold() for path in self.images}
        for path in paths:
            candidate = Path(path)
            if candidate.is_file() and candidate.suffix.lower() in IMAGE_SUFFIXES and (str(candidate.resolve()).casefold() not in existing):
                self.images.append(candidate)
                existing.add(str(candidate.resolve()).casefold())
                self.input_list.insert('end', candidate.name)
        self.input_count.configure(text=tr('text_9f934685f984', p0=f'{len(self.images)}'))

    def _choose_images(self):
        paths = filedialog.askopenfilenames(title=tr('text_020df0a8c65c'), filetypes=FILE_TYPES)
        self._add_images(paths)

    def _choose_folder(self):
        folder = filedialog.askdirectory(title=tr('text_230d8e3e1d62'))
        if folder:
            self._add_images(sorted((path for path in Path(folder).iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)))

    def _drop_images(self, event):
        paths = [Path(path) for path in self.tk.splitlist(event.data)]
        files = []
        for path in paths:
            if path.is_dir():
                files.extend(sorted((item for item in path.iterdir() if item.suffix.lower() in IMAGE_SUFFIXES)))
            else:
                files.append(path)
        self._add_images(files)

    def _clear_images(self):
        if self.running:
            return
        self.images.clear()
        self.input_list.delete(0, 'end')
        self.input_count.configure(text=tr('text_85f2adcfb9db'))

    def _choose_references(self):
        paths = filedialog.askopenfilenames(title=tr('text_51cff1ae322a'), filetypes=FILE_TYPES)
        existing = {str(path.resolve()).casefold() for path in self.references}
        for path in paths:
            candidate = Path(path)
            if candidate.is_file() and str(candidate.resolve()).casefold() not in existing:
                self.references.append(candidate)
                existing.add(str(candidate.resolve()).casefold())
        self.reference_count.configure(text=tr('text_30a97706c37b', p0=f'{len(self.references)}'))

    def _choose_output(self):
        folder = filedialog.askdirectory(title=tr('text_07fbe85a4a18'))
        if folder:
            self.output_folder.set(folder)

    def _open_output(self):
        if self.last_output is None or not self.last_output.is_dir():
            return
        if sys.platform == 'win32':
            os.startfile(self.last_output)
        elif sys.platform == 'darwin':
            subprocess.Popen(['open', str(self.last_output)])
        else:
            subprocess.Popen(['xdg-open', str(self.last_output)])

    def _threshold_changed(self, value):
        self.threshold_label.configure(text=f'{float(value):.2f}')

    def _model_changed(self, event=None):
        spec = model_spec(self.model_id.get())
        self.threshold.set(float(spec['threshold']))
        self.threshold_label.configure(text=f"{float(spec['threshold']):.2f}")
        self.threshold_note.configure(text=tr('text_4711faca156d', p0=f"{float(spec['threshold']):.2f}"))
        self.model_sidebar.configure(text=tr('text_fd91828ca4a3', p0=f"{spec['fold']}"))

    def _start(self):
        if self.running:
            return
        if not self.images:
            messagebox.showinfo(tr('text_cece54fefa88'), tr('text_081eeccc2643'))
            return
        output_text = self.output_folder.get().strip()
        if not output_text:
            messagebox.showerror(tr('text_305c965ad310'), tr('text_a636bb8ab1ae'))
            return
        folder = Path(output_text).expanduser()
        self.running = True
        self.language_choice.configure(state='disabled')
        self.cancel_event.clear()
        self.model_choice.configure(state='disabled')
        self.run_button.configure(state='disabled')
        self.cancel_button.configure(state='normal')
        self.open_output_button.configure(state='disabled')
        self.progress.configure(value=0, maximum=len(self.images))
        self.status.configure(text=tr('text_754cc0e9ad27'))
        self.rows.clear()
        self.current_result = None
        self.table.delete(*self.table.get_children())
        for widget in self.stats.values():
            widget.configure(text='—')
        inputs = list(self.images)
        references = list(self.references)
        threshold = float(self.threshold.get())
        model_id = self.model_id.get()
        save_probability = self.save_probability.get()
        threading.Thread(target=self._worker, args=(inputs, references, folder, threshold, save_probability, model_id), daemon=True).start()

    def _worker(self, images, references, output, threshold, save_probability, model_id):
        set_language(self.language)
        try:
            result = run_batch(images, references, output, threshold, save_probability, self.cancel_event, lambda done, total, text: self.events.put(('progress', done, total, text)), model_id=model_id)
            self.events.put(('complete', result))
        except Exception as error:
            self.events.put(('error', str(error)))

    def _cancel(self):
        if self.running:
            self.cancel_event.set()
            self.status.configure(text=tr('text_d95f4d791673'))
            self.cancel_button.configure(state='disabled')

    def _poll(self):
        try:
            while True:
                event = self.events.get_nowait()
                if event[0] == 'progress':
                    _, done, total, text = event
                    self.progress.configure(maximum=max(total, 1), value=done)
                    self.status.configure(text=text)
                elif event[0] == 'complete':
                    self._complete(event[1])
                elif event[0] == 'error':
                    self.running = False
                    self.language_choice.configure(state='readonly')
                    self.model_choice.configure(state='readonly')
                    self.run_button.configure(state='normal')
                    self.cancel_button.configure(state='disabled')
                    self.status.configure(text=tr('text_127db1eb5e5a'))
                    messagebox.showerror(tr('text_127db1eb5e5a'), event[1])
                    if self.pending_close:
                        self.destroy()
        except queue.Empty:
            pass
        if self.winfo_exists():
            self.after(100, self._poll)

    def _complete(self, result):
        self.running = False
        self.last_result = result
        self.language_choice.configure(state='readonly')
        self.model_choice.configure(state='readonly')
        self.run_button.configure(state='normal')
        self.cancel_button.configure(state='disabled')
        self.last_output = Path(result['folder'])
        self.last_archive = Path(result['archive']) if result['archive'] else None
        self.open_output_button.configure(state='normal')
        rows = result['results']
        errors = result['errors']
        fractions = sorted((item['foreground_fraction'] for item in rows))
        self.stats['images'].configure(text=str(len(rows)))
        self.stats['foreground'].configure(text=f'{100 * ((fractions[(len(fractions) - 1) // 2] + fractions[len(fractions) // 2]) / 2):.2f}' if fractions else '—')
        self.stats['paths'].configure(text=str(sum((item['descriptors']['path_status'] == 'ok' for item in rows))))
        self.stats['qc'].configure(text=str(sum((bool(item['qc_flags']) for item in rows))))
        for item in rows:
            descriptors = item['descriptors']

            def shown(value):
                return '—' if value is None else f'{value:.1f}'
            iid = item['image_id']
            self.table.insert('', 'end', iid=iid, values=(item['image_name'], shown(descriptors['dominant_path_length']), descriptors['retained_segment_count'], descriptors['junction_region_count'], shown(descriptors['mean_local_acute_angle_deg']), tr('text_eaae1132582f') if item['qc_flags'] else tr('text_f78d037abccd')))
            self.rows[iid] = item
        if rows:
            self.table.selection_set(rows[0]['image_id'])
            self.table.focus(rows[0]['image_id'])
            self._select_result()
        suffix = tr('text_2c1999919413', p0=f'{len(errors)}') if errors else ''
        if result['report']['status'] == 'cancelled':
            suffix += tr('text_d1065c1e96d3')
        self.status.configure(text=tr('text_f59a941c28ce', p0=f'{len(rows)}', p1=f'{suffix}', p2=self.last_output.name))
        if not rows and errors:
            messagebox.showwarning(tr('text_77c563bc7f76'), tr('text_1bc4abde15a9', p0=f'{len(errors)}'))
        if self.pending_close:
            self.destroy()

    def _select_result(self, event=None):
        selection = self.table.selection()
        if not selection:
            return
        self.current_result = self.rows.get(selection[0])
        if self.current_result is None:
            return
        item = self.current_result
        self.selected_name.configure(text=item['image_name'])
        descriptors = item['descriptors']
        qc = ', '.join(item['qc_flags']) if item['qc_flags'] else tr('text_72077749f794')
        detail = tr('text_6412c273aa04', p0=f"{item['width']}", p1=f"{item['height']}", p2=f"{descriptors['path_status']}", p3=f"{descriptors['valid_angle_n']}", p4=f'{qc}')
        if 'evaluation' in item:
            evaluation = item['evaluation']

            def metric(value):
                return '—' if value is None else f'{value:.3f}'
            detail += tr('text_eaa8f62504bf', p0=f"{metric(evaluation['dice'])}", p1=f"{metric(evaluation['iou'])}", p2=f"{metric(evaluation['hard_cldice'])}")
        self.details.configure(text=detail)
        self._refresh_preview()

    def _set_layer(self, name):
        self.layer.set(name)
        for key, widget in self.layer_buttons.items():
            widget.configure(bg=WHITE if key == name else '#eff5f2', fg=TEAL if key == name else MUTED)
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
        path = Path(item['_source_path']) if layer is None else self.last_output / 'images' / item['image_id'] / layer
        if not path.exists():
            self.preview.configure(text=tr('text_f598321d61d0'), image='')
            return
        try:
            with Image.open(path) as image:
                picture = image.convert('RGB')
                width = max(100, self.preview.winfo_width() - 18)
                height = max(100, self.preview.winfo_height() - 18)
                picture.thumbnail((width, height), Image.Resampling.LANCZOS)
                self.preview_image = ImageTk.PhotoImage(picture, master=self)
            self.preview.configure(image=self.preview_image, text='')
        except Exception as error:
            self.preview.configure(text=tr('text_f96145384a34', p0=f'{error}'), image='')

    def _close(self):
        if self.running:
            if messagebox.askyesno(tr('text_4e26e96c5481'), tr('text_2381ab93f546')):
                self.pending_close = True
                self._cancel()
            return
        self.destroy()

def main():
    if len(sys.argv) >= 2 and sys.argv[1] == '--batch':
        from cli import main as cli_main
        sys.exit(cli_main(sys.argv[2:]))
    if len(sys.argv) >= 2 and sys.argv[1] == '--self-test':
        import numpy as np
        from rootscope.engine import InferenceEngine
        from rootscope.models import catalog, model_path
        checked = []
        for spec in catalog()['models']:
            engine = InferenceEngine(model_path(spec['id']), spec['sha256'])
            probability = engine.predict(np.zeros((256, 256), dtype=np.uint8))
            if probability.shape != (256, 256) or not np.isfinite(probability).all():
                raise RuntimeError(f"Self-test failed: invalid output for {spec['id']}")
            checked.append({'id': spec['id'], 'sha256': spec['sha256'], 'provider': engine.device})
        report = {'status': 'ok', 'version': __version__, 'models': checked, 'shape': [256, 256]}
        if len(sys.argv) >= 3:
            Path(sys.argv[2]).write_text(json.dumps(report, indent=2), encoding='utf-8')
        else:
            print(json.dumps(report))
        return
    if sys.platform == 'win32':
        try:
            import ctypes
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except (AttributeError, OSError):
            pass
    RootScopeApp().mainloop()
if __name__ == '__main__':
    main()
