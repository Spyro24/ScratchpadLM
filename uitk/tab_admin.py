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


class AdminTabMixin:
    def _build_admin_tab(self, tab):
        """Admin tab: catalog model list, create/delete, catalog mgmt."""
        tab.rowconfigure(1, weight=1)
        tab.columnconfigure(0, weight=1)

        # Catalog list
        list_frame = tk.LabelFrame(tab, text="Model Catalog", padx=10, pady=10)
        list_frame.grid(row=1, column=0, sticky="nsew")
        list_frame.rowconfigure(0, weight=1)
        list_frame.columnconfigure(0, weight=1)

        self.admin_list = tk.Listbox(list_frame, font=("Consolas", 9))
        self.admin_list.grid(row=0, column=0, sticky="nsew")
        scrollbar = tk.Scrollbar(list_frame, orient="vertical", command=self.admin_list.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.admin_list.configure(yscrollcommand=scrollbar.set)

        # Refresh catalog button
        tk.Button(tab, text="Refresh Catalog", command=self.refresh_catalog_list, width=15).grid(row=2, column=0, pady=(8, 0))

        # Create model row
        create_frame = tk.LabelFrame(tab, text="Create Model", padx=10, pady=10)
        create_frame.grid(row=3, column=0, sticky="ew", pady=(10, 0))
        create_frame.columnconfigure(1, weight=1)
        create_frame.columnconfigure(3, weight=1)

        tk.Label(create_frame, text="Name:").grid(row=0, column=0, padx=(0, 5))
        self.create_name_var = tk.Entry(create_frame, width=20)
        self.create_name_var.grid(row=0, column=1, sticky="w", padx=(0, 10))

        tk.Label(create_frame, text="Domains:").grid(row=0, column=2, padx=(0, 5))
        self.create_domains_var = tk.Entry(create_frame, width=30)
        self.create_domains_var.grid(row=0, column=3, sticky="w", padx=(0, 10))

        self.create_btn = tk.Button(create_frame, text="Create", command=self.on_create_model)
        self.create_btn.grid(row=0, column=4, padx=5)

        # Auto name from training data
        analyze_row = tk.Frame(create_frame)
        analyze_row.grid(row=1, column=0, columnspan=5, sticky="ew", pady=(5, 0))
        tk.Label(analyze_row, text="Auto-name from training data:").pack(side="left", padx=(0, 5))
        self.analyze_data_var = tk.Entry(analyze_row, width=40)
        self.analyze_data_var.pack(side="left", padx=5, fill="x", expand=True)
        self.analyze_data_var.insert(0, "Paste training text here to analyze...")
        tk.Button(analyze_row, text="Analyze & Suggest Name", command=self.on_analyze_data).pack(side="left", padx=5)
        self.analyze_result_var = tk.Label(analyze_row, text="", font=("Consolas", 9), fg="#27AE60")
        self.analyze_result_var.pack(side="left", padx=10)

        # Delete model row
        delete_frame = tk.LabelFrame(tab, text="Delete Model", padx=10, pady=10)
        delete_frame.grid(row=4, column=0, sticky="ew", pady=(10, 0))

        self.delete_btn = tk.Button(delete_frame, text="Delete Selected", command=self.on_delete_model)
        self.delete_btn.pack(side="left")
