"""Tkinter UI mixin (stdlib fallback). Part of the `uitk/` package split from `ui.py`."""
import json
import os
import queue
import threading
import time
import tkinter as tk
from tkinter import scrolledtext, messagebox, ttk, filedialog
from pathlib import Path

from tokenizer import BPETokenizer
from ai import TinyLM, train, generate, load_model, save_model, sync_vocab, train_parallel, count_params, Router, RouterTrainingExample
from registry import MODEL_REGISTRY, TASK_REGISTRY, TOKENIZER_MANAGER, get_tokenizer
from model_mgmt import ModelCatalog, ModelCreator, ModelLoader, MODEL_CATALOG
from config import Config
from autotrain import RatingCollector, FeedbackEntry, create_autotrain_system
from ui_base import (
    CONFIG_PATH,
    MODEL_PATH,
    PALETTE,
    PALETTE_LIGHT,
    PALETTE_DARK,
    TOKENIZER_PATH,
    get_available_models,
    load_config,
    apply_tk_theme,
    get_palette,
)
from ui_base import parse_bool as _parse_bool
from ui_base import write_config_yaml as _write_config_yaml
from uitk.const import cfg, VOCAB_SIZE, CTX, EMB, HIDDEN


class TrainingTabMixin:
    def _build_training_tab(self, tab):
        """Training tab: data input, controls, training log monitor."""
        tab.rowconfigure(1, weight=1)  # Training text expands
        tab.rowconfigure(4, weight=1)  # Log monitor expands
        tab.columnconfigure(0, weight=1)

        # Target model indicator (mirrors Chat tab selection)
        target_row = tk.Frame(tab)
        target_row.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        tk.Label(target_row, text="Training target:", font=("Consolas", 9, "bold")).pack(side="left")
        tk.Label(target_row, textvariable=self.model_var, font=("Consolas", 9)).pack(side="left", padx=(5, 0))

        # Training text area
        train_frame = tk.LabelFrame(tab, text="Training Data", padx=10, pady=10)
        train_frame.grid(row=1, column=0, sticky="nsew", pady=(0, 10))
        train_frame.rowconfigure(0, weight=1)
        train_frame.columnconfigure(0, weight=1)

        self.train_text = scrolledtext.ScrolledText(train_frame, wrap="word", height=6)
        self.train_text.grid(row=0, column=0, sticky="nsew")
        self.train_text.insert("1.0", "Enter training text here...")

        # Training controls row
        train_controls = tk.Frame(tab)
        train_controls.grid(row=2, column=0, sticky="ew", pady=(0, 10))

        tk.Label(train_controls, text="Steps:").pack(side="left", padx=(0, 5))
        self.steps_var = tk.Entry(train_controls, width=10)
        self.steps_var.insert(0, str(self.cfg["training"]["steps"]))
        self.steps_var.pack(side="left", padx=(0, 10))

        self.train_btn = tk.Button(train_controls, text="Train on Data", width=15, command=self.on_train)
        self.train_btn.pack(side="left", padx=5)

        self.train_router_btn = tk.Button(train_controls, text="Train Router", width=15, command=self.on_train_router)
        self.train_router_btn.pack(side="left", padx=5)

        self.cancel_train_btn = tk.Button(train_controls, text="Cancel Training", width=15, command=self.on_cancel_training, state="disabled")
        self.cancel_train_btn.pack(side="left", padx=5)

        # Progress bar row
        progress_row = tk.Frame(tab)
        progress_row.grid(row=3, column=0, sticky="ew", pady=(8, 0))
        progress_row.columnconfigure(0, weight=1)
        progress_row.columnconfigure(1, weight=1)

        self.train_progress = ttk.Progressbar(progress_row, mode="determinate", maximum=100)
        self.train_progress.grid(row=0, column=0, sticky="ew", padx=(0, 8))
        self.progress_label = tk.Label(progress_row, text="", font=("Consolas", 9), anchor="w")
        self.progress_label.grid(row=0, column=1, sticky="ew")

        # Training monitor (log)
        monitor_frame = tk.LabelFrame(tab, text="Training Monitor", padx=10, pady=10)
        monitor_frame.grid(row=4, column=0, sticky="nsew")
        monitor_frame.rowconfigure(0, weight=1)
        monitor_frame.columnconfigure(0, weight=1)

        self.train_log = scrolledtext.ScrolledText(
            monitor_frame, wrap="word", state="disabled", height=8, font=("Consolas", 9)
        )
        self.train_log.grid(row=0, column=0, sticky="nsew")
        self.train_log.tag_configure("ts", foreground=PALETTE["muted"])
        self.train_log.tag_configure("err", foreground=PALETTE["error"])
