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
from tinylm import TinyLM, train, generate, load_model, save_model, sync_vocab, train_parallel, count_params, Router, RouterTrainingExample
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


class InfoTabMixin:
    def _build_model_info_tab(self, tab):
        """Model Info tab: inspector + tensor details."""
        self.inspector_frame = tk.LabelFrame(tab, text="Model Inspector", padx=10, pady=10)
        self.inspector_frame.pack(fill="x")
        self.inspector_frame.columnconfigure(1, weight=1)
        self.inspector_frame.columnconfigure(3, weight=1)

        # Model name
        tk.Label(self.inspector_frame, text="Model:", font=("Consolas", 9, "bold")).grid(row=0, column=0, sticky="w", padx=(0, 5))
        self.inspector_model_name = tk.Label(self.inspector_frame, text="None", font=("Consolas", 9))
        self.inspector_model_name.grid(row=0, column=1, sticky="w", padx=(0, 20))

        # Architecture
        tk.Label(self.inspector_frame, text="Architecture:", font=("Consolas", 9, "bold")).grid(row=0, column=2, sticky="w", padx=(0, 5))
        self.inspector_arch = tk.Label(self.inspector_frame, text="Unknown", font=("Consolas", 9))
        self.inspector_arch.grid(row=0, column=3, sticky="w")

        # Parameter count
        tk.Label(self.inspector_frame, text="Parameters:", font=("Consolas", 9, "bold")).grid(row=1, column=0, sticky="w", padx=(0, 5), pady=(5, 0))
        self.inspector_params = tk.Label(self.inspector_frame, text="0", font=("Consolas", 9))
        self.inspector_params.grid(row=1, column=1, sticky="w", padx=(0, 20), pady=(5, 0))

        # Memory estimate (4-bit quantized)
        tk.Label(self.inspector_frame, text="Memory (4-bit):", font=("Consolas", 9, "bold")).grid(row=1, column=2, sticky="w", padx=(0, 5), pady=(5, 0))
        self.inspector_memory = tk.Label(self.inspector_frame, text="0 KB", font=("Consolas", 9))
        self.inspector_memory.grid(row=1, column=3, sticky="w", pady=(5, 0))

        # Vocab size
        tk.Label(self.inspector_frame, text="Vocab Size:", font=("Consolas", 9, "bold")).grid(row=2, column=0, sticky="w", padx=(0, 5), pady=(5, 0))
        self.inspector_vocab = tk.Label(self.inspector_frame, text="0", font=("Consolas", 9))
        self.inspector_vocab.grid(row=2, column=1, sticky="w", padx=(0, 20), pady=(5, 0))

        # Context size
        tk.Label(self.inspector_frame, text="Context:", font=("Consolas", 9, "bold")).grid(row=2, column=2, sticky="w", padx=(0, 5), pady=(5, 0))
        self.inspector_ctx = tk.Label(self.inspector_frame, text="0", font=("Consolas", 9))
        self.inspector_ctx.grid(row=2, column=3, sticky="w", pady=(5, 0))

        # Embedding dim
        tk.Label(self.inspector_frame, text="Embedding:", font=("Consolas", 9, "bold")).grid(row=3, column=0, sticky="w", padx=(0, 5), pady=(5, 0))
        self.inspector_emb = tk.Label(self.inspector_frame, text="0", font=("Consolas", 9))
        self.inspector_emb.grid(row=3, column=1, sticky="w", padx=(0, 20), pady=(5, 0))

        # Hidden size
        tk.Label(self.inspector_frame, text="Hidden:", font=("Consolas", 9, "bold")).grid(row=3, column=2, sticky="w", padx=(0, 5), pady=(5, 0))
        self.inspector_hidden = tk.Label(self.inspector_frame, text="0", font=("Consolas", 9))
        self.inspector_hidden.grid(row=3, column=3, sticky="w", pady=(5, 0))

        # Buttons: tensor details popup + manual refresh
        btn_row = tk.Frame(tab)
        btn_row.pack(pady=(10, 0))
        self.tensor_detail_btn = tk.Button(btn_row, text="Show Tensor Details", command=self.on_show_tensor_details, state="disabled")
        self.tensor_detail_btn.pack(side="left", padx=5)
        tk.Button(btn_row, text="Refresh", command=self.update_model_inspector).pack(side="left", padx=5)
