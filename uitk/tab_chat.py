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


class ChatTabMixin:
    def _build_chat_tab(self, tab):
        """Chat tab: model selector, conversation history, prompt input."""
        tab.columnconfigure(0, weight=1)

        # Model selector row
        model_row = tk.Frame(tab)
        model_row.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        model_row.columnconfigure(1, weight=1)

        tk.Label(model_row, text="Model:").grid(row=0, column=0, padx=(0, 5))
        self.model_var = tk.StringVar(value="TinyLM")
        self.model_combo = ttk.Combobox(model_row, textvariable=self.model_var, state="readonly", width=20)
        self.model_combo["values"] = get_available_models()
        self.model_combo.grid(row=0, column=1, sticky="w", padx=(0, 10))
        self.model_combo.bind("<<ComboboxSelected>>", self.on_model_change)

        # Refresh models button
        tk.Button(model_row, text="Refresh Models", command=self.refresh_models, width=15).grid(row=0, column=2, padx=5)

        # Use selected model instead of router
        self.use_selected_model = tk.BooleanVar(value=False)
        tk.Checkbutton(model_row, text="Use selected model (bypass router)", variable=self.use_selected_model).grid(row=0, column=3, padx=10, sticky="w")

        # Vocab size display
        self.vocab_label = tk.Label(model_row, text=f"Vocab: {len(self.tok.vocab)}")
        self.vocab_label.grid(row=0, column=4, padx=10)

        # Conversation history
        history_frame = tk.LabelFrame(tab, text="Conversation", padx=10, pady=10)
        history_frame.grid(row=1, column=0, sticky="nsew", pady=(0, 10))
        history_frame.rowconfigure(0, weight=1)
        history_frame.columnconfigure(0, weight=1)

        self.history = scrolledtext.ScrolledText(
            history_frame, wrap="word", state="disabled", font=("Consolas", 10)
        )
        self.history.grid(row=0, column=0, sticky="nsew")

        # Configure tags for styling (colors from shared ui_base.PALETTE,
        # identical to the PySide6 theme)
        self.history.tag_configure("user", foreground=PALETTE["accent"], font=("Consolas", 10, "bold"))
        self.history.tag_configure("assistant", foreground=PALETTE["success"], font=("Consolas", 10, "bold"))
        self.history.tag_configure("system", foreground=PALETTE["muted"], font=("Consolas", 9, "italic"))
        self.history.tag_configure("error", foreground=PALETTE["error"], font=("Consolas", 10, "bold"))

        # Prompt input
        prompt_frame = tk.LabelFrame(tab, text="Prompt", padx=10, pady=10)
        prompt_frame.grid(row=2, column=0, sticky="ew")
        prompt_frame.columnconfigure(0, weight=1)

        prompt_row = tk.Frame(prompt_frame)
        prompt_row.grid(row=0, column=0, sticky="ew")
        prompt_row.columnconfigure(0, weight=1)

        self.prompt_var = tk.Entry(prompt_row, font=("Consolas", 11))
        self.prompt_var.grid(row=0, column=0, sticky="ew", padx=(0, 8))
        self.prompt_var.bind("<Return>", lambda e: self.on_send_prompt())
        self.prompt_var.focus_set()

        self.send_btn = tk.Button(prompt_row, text="Send", width=10, command=self.on_send_prompt)
        self.send_btn.grid(row=0, column=1)

        # Quick generation settings - use grid with wrapping
        gen_frame = tk.LabelFrame(prompt_frame, text="Generation Settings", padx=10, pady=8)
        gen_frame.grid(row=1, column=0, sticky="ew", pady=(8, 0))
        gen_frame.columnconfigure(1, weight=1)
        gen_frame.columnconfigure(3, weight=1)
        gen_frame.columnconfigure(5, weight=1)
        gen_frame.columnconfigure(7, weight=1)

        # Row 0: Max Tokens, Min Tokens, Temp, Top-k, Top-p
        settings_row1 = [
            ("Max Tokens (0=∞):", "max_tokens_var", "generation", "max_tokens", 8),
            ("Min Tokens:", "min_tokens_var", "generation", "min_tokens", 8),
            ("Temp:", "temp_var", "generation", "temperature", 8),
            ("Top-k:", "top_k_var", "generation", "top_k", 8),
            ("Top-p:", "top_p_var", "generation", "top_p", 8),
        ]
        for i, (label, attr, section, key, width) in enumerate(settings_row1):
            col = i * 2
            tk.Label(gen_frame, text=label).grid(row=0, column=col, padx=(0, 5), sticky="w")
            entry = tk.Entry(gen_frame, width=width)
            entry.insert(0, str(self.cfg[section][key]))
            entry.grid(row=0, column=col+1, padx=(0, 15), sticky="ew")
            setattr(self, attr, entry)

        # Row 1: Repetition penalty, Stop tokens, Stop strings
        tk.Label(gen_frame, text="Rep. Penalty:").grid(row=1, column=0, padx=(0, 5), sticky="w", pady=(4, 0))
        self.rep_penalty_var = tk.Entry(gen_frame, width=8)
        self.rep_penalty_var.insert(0, str(self.cfg["generation"]["repetition_penalty"]))
        self.rep_penalty_var.grid(row=1, column=1, padx=(0, 15), sticky="ew", pady=(4, 0))

        tk.Label(gen_frame, text="Stop IDs:").grid(row=1, column=2, padx=(0, 5), sticky="w", pady=(4, 0))
        self.stop_tokens_var = tk.Entry(gen_frame, width=15)
        self.stop_tokens_var.insert(0, ",".join(str(x) for x in self.cfg["generation"].get("stop_token_ids", [])))
        self.stop_tokens_var.grid(row=1, column=3, columnspan=2, padx=(0, 15), sticky="ew", pady=(4, 0))

        tk.Label(gen_frame, text="Stop Strings:").grid(row=1, column=5, padx=(0, 5), sticky="w", pady=(4, 0))
        self.stop_strings_var = tk.Entry(gen_frame, width=15)
        self.stop_strings_var.insert(0, ",".join(self.cfg["generation"].get("stop_strings", [])))
        self.stop_strings_var.grid(row=1, column=6, columnspan=2, padx=(0, 15), sticky="ew", pady=(4, 0))

        self.clear_btn = tk.Button(gen_frame, text="Clear History", command=self.on_clear_history)
        self.clear_btn.grid(row=2, column=0, columnspan=3, sticky="w", pady=(4, 0), padx=10)
        
        self.preview_btn = tk.Button(gen_frame, text="Generate with Preview", command=self.on_generate_preview)
        self.preview_btn.grid(row=2, column=3, columnspan=3, sticky="w", pady=(4, 0), padx=10)
