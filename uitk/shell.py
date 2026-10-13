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


class ShellMixin:
    def _build_ui(self):
        # Main container with padding
        main = tk.Frame(self.root, padx=10, pady=10)
        main.pack(fill="both", expand=True)
        main.rowconfigure(0, weight=1)
        main.columnconfigure(0, weight=1)

        # Tabbed layout (roadmap 0a: Chat, Training, Model Info, Admin, Settings, Logs)
        self.notebook = ttk.Notebook(main)
        self.notebook.grid(row=0, column=0, sticky="nsew")

        # Create scrollable tabs for tabs with lots of content
        chat_tab = self._create_scrollable_tab(self.notebook, "Chat")
        training_tab = self._create_scrollable_tab(self.notebook, "Training")
        info_tab = self._create_scrollable_tab(self.notebook, "Model Info")
        admin_tab = self._create_scrollable_tab(self.notebook, "Admin")
        settings_tab = self._create_scrollable_tab(self.notebook, "Settings")
        logs_tab = self._create_scrollable_tab(self.notebook, "Logs")

        self._build_chat_tab(chat_tab)
        self._build_training_tab(training_tab)
        self._build_model_info_tab(info_tab)
        self._build_admin_tab(admin_tab)
        self._build_settings_tab(settings_tab)
        self._build_logs_tab(logs_tab)

        # Status bar at bottom
        self.status = tk.Label(self.root, text="Ready", anchor="w", bd=1, relief="sunken", padx=5)
        self.status.pack(fill="x", side="bottom", padx=10, pady=(0, 10))

    def _create_scrollable_tab(self, notebook, title):
        """Create a scrollable tab frame."""
        # Outer frame for the tab
        tab_frame = tk.Frame(notebook)
        notebook.add(tab_frame, text=title)
        tab_frame.rowconfigure(0, weight=1)
        tab_frame.columnconfigure(0, weight=1)

        # Canvas with scrollbar
        canvas = tk.Canvas(tab_frame, highlightthickness=0)
        scrollbar = ttk.Scrollbar(tab_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, padx=10, pady=10)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")

        # Bind mousewheel
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)
        canvas.bind_all("<Button-4>", lambda e: canvas.yview_scroll(-1, "units"))
        canvas.bind_all("<Button-5>", lambda e: canvas.yview_scroll(1, "units"))

        return scrollable_frame
