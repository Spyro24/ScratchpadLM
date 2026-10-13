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


class SettingsTabMixin:
    def _build_settings_tab(self, tab):
        """Settings tab: edit generation/training/tokenizer/model config, save to config.yaml.
        Includes Tokenizer Manager section for viewing/extending vocabulary."""
        tab.columnconfigure(1, weight=1)

        # (section, key, label, type) for each editable setting
        settings = [
            ("generation", "max_tokens", "Max tokens", int),
            ("generation", "min_tokens", "Min tokens", int),
            ("generation", "temperature", "Temperature", float),
            ("generation", "top_k", "Top-k", int),
            ("generation", "top_p", "Top-p", float),
            ("generation", "repetition_penalty", "Repetition penalty", float),
            ("generation", "stop_token_ids", "Stop token IDs (comma-separated)", str),
            ("generation", "stop_strings", "Stop strings (comma-separated)", str),
            ("training", "steps", "Training steps", int),
            ("training", "lr", "Learning rate", float),
            ("training", "batch_size", "Batch size", int),
            ("training", "n_negatives", "Negatives", int),
            ("training", "workers", "Workers", int),
            ("training", "lr_scheduler", "LR scheduler", str),
            ("training", "warmup_steps", "Warmup steps", int),
            ("training", "min_lr", "Min learning rate", float),
            ("training", "grad_clip", "Grad clip norm", float),
            ("training", "val_split", "Validation split", float),
            ("training", "eval_every", "Eval every N steps", int),
            ("training", "early_stopping", "Early stopping", _parse_bool),
            ("training", "patience", "Patience", int),
            ("training", "min_delta", "Min delta", float),
            ("training", "checkpoint_every", "Checkpoint every N", int),
            ("training", "checkpoint_dir", "Checkpoint dir", str),
            ("training", "auto_adjust", "Auto LR adjust", _parse_bool),
            ("training", "auto_factor", "Auto LR factor", float),
            ("training", "auto_patience", "Auto LR patience", int),
            ("training", "auto_clip_relax", "Auto clip relax", _parse_bool),
            ("tokenizer", "vocab_size", "Tokenizer vocab size", int),
            ("tokenizer", "min_frequency", "Min frequency", int),
            ("model", "vocab_size", "Model vocab size", int),
            ("model", "ctx", "Context length", int),
            ("model", "emb", "Embedding dim", int),
            ("model", "hidden", "Hidden dim", int),
            ("ui", "theme", "UI Theme", str),
        ]

        self._settings_entries = {}  # (section, key) -> (Entry, type)

        # Use two-column layout for settings
        tab.columnconfigure(1, weight=1)
        tab.columnconfigure(3, weight=1)

        row = 0
        col = 0
        current_section = None
        for idx, (section, key, label, cast) in enumerate(settings):
            # Switch to right column halfway through
            if idx == len(settings) // 2 + 1:
                col = 2
                row = 0

            if section != current_section:
                current_section = section
                tk.Label(tab, text=section.upper(), font=("Consolas", 10, "bold")).grid(
                    row=row, column=col, columnspan=2, sticky="w", pady=(10 if row else 0, 2)
                )
                row += 1
            
            # Special handling for theme setting - use Combobox
            if section == "ui" and key == "theme":
                tk.Label(tab, text=label + ":").grid(row=row, column=col, sticky="w", padx=(10, 5))
                self.theme_var = tk.StringVar(value=str(self.cfg[section][key]))
                theme_combo = ttk.Combobox(tab, textvariable=self.theme_var, values=["light", "dark", "system"], state="readonly", width=10)
                theme_combo.grid(row=row, column=col + 1, sticky="ew", pady=1)
                theme_combo.bind("<<ComboboxSelected>>", self._on_theme_change)
                # Store with a dummy cast for consistency
                self._settings_entries[(section, key)] = (theme_combo, cast)
            else:
                tk.Label(tab, text=label + ":").grid(row=row, column=col, sticky="w", padx=(10, 5))
                entry = tk.Entry(tab, width=12)
                entry.insert(0, str(self.cfg[section][key]))
                entry.grid(row=row, column=col + 1, sticky="ew", pady=1)
                self._settings_entries[(section, key)] = (entry, cast)
            row += 1

        # ---- Tokenizer Manager Section (spans both columns) ----
        tokenizer_row = max(row, len(settings) // 2 + 5)  # Start after the longest column
        tk.Label(tab, text="TOKENIZER MANAGER", font=("Consolas", 10, "bold")).grid(
            row=tokenizer_row, column=0, columnspan=4, sticky="w", pady=(15, 2)
        )
        tokenizer_row += 1

        # Vocab size display
        tk.Label(tab, text="Current Vocab Size:").grid(row=tokenizer_row, column=0, sticky="w", padx=(10, 5))
        self.tok_vocab_label = tk.Label(tab, text=str(len(self.tok.vocab)), font=("Consolas", 9, "bold"))
        self.tok_vocab_label.grid(row=tokenizer_row, column=1, sticky="w", pady=1)
        tokenizer_row += 1

        # Number of learned merges
        tk.Label(tab, text="Learned Merges:").grid(row=tokenizer_row, column=0, sticky="w", padx=(10, 5))
        self.tok_merges_label = tk.Label(tab, text=str(len(self.tok.merges)), font=("Consolas", 9, "bold"))
        self.tok_merges_label.grid(row=tokenizer_row, column=1, sticky="w", pady=1)
        tokenizer_row += 1

        # Extend vocab text area
        tk.Label(tab, text="Extend with text:").grid(row=tokenizer_row, column=0, sticky="nw", padx=(10, 5))
        self.tok_extend_text = scrolledtext.ScrolledText(tab, wrap="word", width=50, height=4, font=("Consolas", 9))
        self.tok_extend_text.grid(row=tokenizer_row, column=1, columnspan=3, sticky="ew", pady=2)
        tokenizer_row += 1

        # Extend vocab controls
        extend_row = tk.Frame(tab)
        extend_row.grid(row=tokenizer_row, column=1, columnspan=3, sticky="w", pady=2)
        tk.Label(extend_row, text="Target vocab size:").pack(side="left", padx=(0, 5))
        self.tok_target_vocab_var = tk.Entry(extend_row, width=10)
        self.tok_target_vocab_var.insert(0, str(self.cfg["tokenizer"]["vocab_size"]))
        self.tok_target_vocab_var.pack(side="left", padx=(0, 10))
        tk.Label(extend_row, text="Min freq:").pack(side="left", padx=(0, 5))
        self.tok_min_freq_var = tk.Entry(extend_row, width=8)
        self.tok_min_freq_var.insert(0, str(self.cfg["tokenizer"]["min_frequency"]))
        self.tok_min_freq_var.pack(side="left", padx=(0, 10))
        tk.Button(extend_row, text="Extend Vocab", command=self.on_extend_vocab).pack(side="left", padx=5)
        tokenizer_row += 1

        # Learned patterns viewer
        tk.Label(tab, text="Learned Patterns (sample):").grid(row=tokenizer_row, column=0, sticky="nw", padx=(10, 5))
        self.tok_patterns_text = scrolledtext.ScrolledText(tab, wrap="word", width=50, height=8, font=("Consolas", 9), state="disabled")
        self.tok_patterns_text.grid(row=tokenizer_row, column=1, columnspan=3, sticky="ew", pady=2)
        tokenizer_row += 1

        tk.Button(tab, text="Refresh Patterns", command=self.refresh_tokenizer_patterns).grid(
            row=tokenizer_row, column=1, sticky="w", pady=(2, 0)
        )
        tokenizer_row += 1

        # ---- Config Editor Section ----
        tokenizer_row += 1
        tk.Label(tab, text="CONFIG EDITOR", font=("Consolas", 10, "bold")).grid(
            row=tokenizer_row, column=0, columnspan=4, sticky="w", pady=(15, 2)
        )
        tokenizer_row += 1

        self.config_editor = scrolledtext.ScrolledText(tab, wrap="word", width=80, height=15, font=("Consolas", 9))
        self.config_editor.grid(row=tokenizer_row, column=0, columnspan=4, sticky="ew", pady=2)
        tokenizer_row += 1

        config_btn_row = tk.Frame(tab)
        config_btn_row.grid(row=tokenizer_row, column=0, columnspan=4, sticky="w", pady=2)
        tk.Button(config_btn_row, text="Load Current Config", command=self.on_load_config_editor).pack(side="left", padx=5)
        tk.Button(config_btn_row, text="Validate Config", command=self.on_validate_config_editor).pack(side="left", padx=5)
        tk.Button(config_btn_row, text="Save Config from Editor", command=self.on_save_config_editor).pack(side="left", padx=5)
        tokenizer_row += 1

        # Preset management
        preset_row = tk.Frame(tab)
        preset_row.grid(row=tokenizer_row, column=0, columnspan=4, sticky="w", pady=2)
        tk.Label(preset_row, text="Preset:").pack(side="left", padx=(0, 5))
        self.preset_var = tk.StringVar(value="Default")
        preset_combo = ttk.Combobox(preset_row, textvariable=self.preset_var, values=["Default", "Fast Training", "High Quality", "Low Memory", "Custom..."], state="readonly", width=15)
        preset_combo.pack(side="left", padx=5)
        preset_combo.bind("<<ComboboxSelected>>", self.on_preset_changed)
        tk.Button(preset_row, text="Save as Preset", command=self.on_save_preset).pack(side="left", padx=5)
        tokenizer_row += 1

        tk.Button(tab, text="Save Config", command=self.on_save_config, width=15).grid(
            row=tokenizer_row, column=0, columnspan=4, pady=(15, 0)
        )

        # Initial pattern refresh
        self.refresh_tokenizer_patterns()
