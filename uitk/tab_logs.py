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


class LogsTabMixin:
    def _build_logs_tab(self, tab):
        """Logs tab: unified log viewer for training, generation, system, and error logs with filtering."""
        tab.rowconfigure(1, weight=1)
        tab.columnconfigure(0, weight=1)

        # Filter controls
        filter_frame = tk.LabelFrame(tab, text="Filters", padx=10, pady=8)
        filter_frame.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        filter_frame.columnconfigure(1, weight=1)

        # Log type checkboxes
        self.log_filter_vars = {}
        log_types = [
            ("training", "Training", "#2E86C1"),
            ("generation", "Generation", "#27AE60"),
            ("system", "System", "#7F8C8D"),
            ("error", "Error", "#E74C3C"),
        ]
        for i, (key, label, color) in enumerate(log_types):
            var = tk.BooleanVar(value=True)
            self.log_filter_vars[key] = var
            cb = tk.Checkbutton(filter_frame, text=label, variable=var, 
                                command=self._apply_log_filters, fg=color, font=("Consolas", 9))
            cb.grid(row=0, column=i, padx=8, sticky="w")

        # Search/filter entry
        tk.Label(filter_frame, text="Search:").grid(row=0, column=len(log_types), padx=(20, 5))
        self.log_search_var = tk.Entry(filter_frame, width=20)
        self.log_search_var.grid(row=0, column=len(log_types)+1, padx=(0, 10))
        self.log_search_var.bind("<KeyRelease>", lambda e: self._apply_log_filters())

        # Auto-scroll checkbox
        self.log_autoscroll_var = tk.BooleanVar(value=True)
        tk.Checkbutton(filter_frame, text="Auto-scroll", variable=self.log_autoscroll_var,
                       font=("Consolas", 9)).grid(row=0, column=len(log_types)+2, padx=8)

        # Clear and Export buttons
        tk.Button(filter_frame, text="Clear Logs", command=self._clear_unified_logs, width=12).grid(
            row=0, column=len(log_types)+3, padx=5)
        tk.Button(filter_frame, text="Export Logs", command=self._export_logs, width=12).grid(
            row=0, column=len(log_types)+4, padx=5)

        # Unified log display
        log_frame = tk.LabelFrame(tab, text="Unified Logs", padx=10, pady=10)
        log_frame.grid(row=1, column=0, sticky="nsew")
        log_frame.rowconfigure(0, weight=1)
        log_frame.columnconfigure(0, weight=1)

        self.unified_log = scrolledtext.ScrolledText(
            log_frame, wrap="word", state="disabled", font=("Consolas", 9)
        )
        self.unified_log.grid(row=0, column=0, sticky="nsew")

        # Configure tags for different log types
        self.unified_log.tag_configure("training", foreground=PALETTE["accent"])
        self.unified_log.tag_configure("generation", foreground=PALETTE["success"])
        self.unified_log.tag_configure("system", foreground=PALETTE["muted"], font=("Consolas", 9, "italic"))
        self.unified_log.tag_configure("error", foreground=PALETTE["error"], font=("Consolas", 9, "bold"))
        self.unified_log.tag_configure("timestamp", foreground="#95A5A6")
        self.unified_log.tag_configure("hidden", elide=True)  # For filtering

        # Store all log entries for filtering
        self._all_log_entries = []  # List of (timestamp, log_type, message)
