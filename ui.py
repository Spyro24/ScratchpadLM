"""Tkinter (stdlib fallback) GUI for ScratchpadLM.

PySide6 (Qt6) is the primary toolkit — see ``ui_qt.py``. Both toolkits are
launched through ``ui_base.run_gui()``, which auto-detects the best toolkit
at startup and falls back to Tkinter when Qt is unavailable, so the project
remains usable with the standard library alone. Force a toolkit with the
``SCRATCHPADLM_UI`` environment variable (``pyside6`` or ``tkinter``).
"""

import os
import queue
import threading
import time
import tkinter as tk
from tkinter import scrolledtext, messagebox, ttk, filedialog
from pathlib import Path
import json

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
    TOKENIZER_PATH,
    get_available_models,
    load_config,
)
from ui_base import parse_bool as _parse_bool
from ui_base import write_config_yaml as _write_config_yaml

# Load configuration
cfg = load_config(CONFIG_PATH)

VOCAB_SIZE = cfg["model"]["vocab_size"]
CTX = cfg["model"]["ctx"]
EMB = cfg["model"]["emb"]
HIDDEN = cfg["model"]["hidden"]


class App:
    def __init__(self, root):
        self.root = root
        root.title("ScratchpadLM (TinyLM)")
        root.geometry("900x720")

        self.queue = queue.Queue()
        self.busy = False
        self.cfg = cfg  # Store config reference

        # Use shared tokenizer manager
        self.tok = TOKENIZER_MANAGER.tokenizer
        if os.path.exists(TOKENIZER_PATH):
            TOKENIZER_MANAGER.load(TOKENIZER_PATH)
        self.model = self._load_model()
        self.router = self._load_router()  # Load Router model

        # Rating collector for auto-training feedback
        self.rating_collector = RatingCollector()
        self._last_assistant_response = None  # (prompt, response, model_name)
        self._stream_active = False  # True while streamed tokens land in history

        self._catalog_names = []  # names shown in Admin listbox (parallel order)

        self._build_ui()
        self._model_name = self.model_var.get()  # main-thread snapshot for worker jobs
        self.update_model_inspector()
        self.refresh_catalog_list()
        self.root.after(100, self._poll_queue)

    def _load_router(self):
        """Load or create Router model."""
        router_path = "router.tlm"
        try:
            router = load_model(router_path)
            if len(router.C) == len(self.tok.vocab):
                return router
        except Exception:
            pass
        # Create new Router with current vocab size
        from ai import Router, RouterConfig
        config = RouterConfig(
            vocab_size=len(self.tok.vocab),
            ctx=CTX,
            emb=EMB,
            hidden=HIDDEN,
            n_layers=2,
            d_ff=512
        )
        return Router(config, seed=42)

    # ---- setup -------------------------------------------------------------

    def _load_model(self):
        """Load the saved model if it matches the tokenizer, else create a new one."""
        try:
            model = load_model(MODEL_PATH)
            if len(model.C) == len(self.tok.vocab):
                return model
        except Exception:
            pass
        return TinyLM(vocab_size=len(self.tok.vocab), ctx=CTX, emb=EMB, hidden=HIDDEN)

    def refresh_models(self):
        """Refresh the model dropdown with available models."""
        models = get_available_models()
        self.model_combo["values"] = models
        if self.model_var.get() not in models:
            self.model_var.set(models[0] if models else "TinyLM")
        self._model_name = self.model_var.get()
        self._append_history("system", f"[Refreshed model list: {', '.join(models)}]")

    def on_model_change(self, event=None):
        """Handle model selection change."""
        model_name = self.model_var.get()
        self._model_name = model_name  # main-thread snapshot for worker jobs
        self._switch_model(model_name)

    def _switch_model(self, model_name):
        """Switch to the selected model."""
        if self.busy:
            return
        self._append_history("system", f"[Switching to model: {model_name}...]")
        
        # Run model loading in background
        self._run_in_background(f"Loading {model_name}", self._load_model_job, model_name)

    def _load_model_job(self, model_name):
        """Background job to load a model."""
        if model_name == "TinyLM":
            # Create a fresh TinyLM with current vocab size
            return TinyLM(vocab_size=len(self.tok.vocab), ctx=CTX, emb=EMB, hidden=HIDDEN)
        try:
            factory = MODEL_REGISTRY.get(model_name)
            return factory()
        except KeyError:
            # Try loading from catalog
            try:
                loader = ModelLoader(MODEL_CATALOG, tokenizer_manager=TOKENIZER_MANAGER)
                return loader.load_model(model_name)
            except Exception as e:
                raise ValueError(f"Model '{model_name}' not found in registry or catalog: {e}")

    def _get_model_factory(self, model_name):
        """Get model factory from registry or return TinyLM."""
        if model_name == "TinyLM":
            return lambda: TinyLM(vocab_size=len(self.tok.vocab), ctx=CTX, emb=EMB, hidden=HIDDEN)
        try:
            return MODEL_REGISTRY.get(model_name)
        except KeyError:
            return None

    # ---- UI ----------------------------------------------------------------

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

        # Vocab size display
        self.vocab_label = tk.Label(model_row, text=f"Vocab: {len(self.tok.vocab)}")
        self.vocab_label.grid(row=0, column=3, padx=10)

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

        # Row 0: Tokens, Temp, Top-k, Top-p
        settings_row1 = [
            ("Tokens:", "n_tokens_var", "generation", "n_tokens", 8),
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

        # Row 1: Repetition penalty, Clear button
        tk.Label(gen_frame, text="Rep. Penalty:").grid(row=1, column=0, padx=(0, 5), sticky="w", pady=(4, 0))
        self.rep_penalty_var = tk.Entry(gen_frame, width=8)
        self.rep_penalty_var.insert(0, str(self.cfg["generation"]["repetition_penalty"]))
        self.rep_penalty_var.grid(row=1, column=1, padx=(0, 15), sticky="ew", pady=(4, 0))

        self.clear_btn = tk.Button(gen_frame, text="Clear History", command=self.on_clear_history)
        self.clear_btn.grid(row=1, column=2, columnspan=3, sticky="w", pady=(4, 0), padx=10)
        
        self.preview_btn = tk.Button(gen_frame, text="Generate with Preview", command=self.on_generate_preview)
        self.preview_btn.grid(row=1, column=5, columnspan=3, sticky="w", pady=(4, 0), padx=10)

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

        tk.Label(create_frame, text="Name:").grid(row=0, column=0, padx=(0, 5))
        self.create_name_var = tk.Entry(create_frame, width=20)
        self.create_name_var.grid(row=0, column=1, sticky="w", padx=(0, 10))

        tk.Label(create_frame, text="Domains:").grid(row=0, column=2, padx=(0, 5))
        self.create_domains_var = tk.Entry(create_frame, width=30)
        self.create_domains_var.grid(row=0, column=3, sticky="w", padx=(0, 10))

        self.create_btn = tk.Button(create_frame, text="Create", command=self.on_create_model)
        self.create_btn.grid(row=0, column=4)

        # Delete model row
        delete_frame = tk.LabelFrame(tab, text="Delete Model", padx=10, pady=10)
        delete_frame.grid(row=4, column=0, sticky="ew", pady=(10, 0))

        self.delete_btn = tk.Button(delete_frame, text="Delete Selected", command=self.on_delete_model)
        self.delete_btn.pack(side="left")

    def _build_settings_tab(self, tab):
        """Settings tab: edit generation/training/tokenizer/model config, save to config.yaml.
        Includes Tokenizer Manager section for viewing/extending vocabulary."""
        tab.columnconfigure(1, weight=1)

        # (section, key, label, type) for each editable setting
        settings = [
            ("generation", "n_tokens", "Generation tokens", int),
            ("generation", "temperature", "Temperature", float),
            ("generation", "top_k", "Top-k", int),
            ("generation", "top_p", "Top-p", float),
            ("generation", "repetition_penalty", "Repetition penalty", float),
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

        tk.Button(tab, text="Save Config", command=self.on_save_config, width=15).grid(
            row=tokenizer_row, column=0, columnspan=4, pady=(15, 0)
        )

        # Initial pattern refresh
        self.refresh_tokenizer_patterns()

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

    def _append_history(self, role, text):
        """Append a message to conversation history with role styling."""
        self.history.configure(state="normal")
        
        if role == "user":
            self.history.insert("end", "You: ", "user")
            self.history.insert("end", text + "\n\n")
            # Store user prompt for rating the next response
            self._last_user_prompt = text
        elif role == "assistant":
            self.history.insert("end", "AI: ", "assistant")
            self.history.insert("end", text + "\n\n")
            # Store for rating
            self._last_assistant_response = (getattr(self, '_last_user_prompt', ''), text, self.model_var.get())
            # Add rating buttons after the response
            self._add_rating_buttons(text)
        elif role == "system":
            self.history.insert("end", text + "\n\n", "system")
        elif role == "error":
            self.history.insert("end", text + "\n\n", "error")
        
        self.history.see("end")
        self.history.configure(state="disabled")
        
        # Also log to unified log viewer
        self._log_unified("system" if role != "error" else "error", text)

    def _add_rating_buttons(self, response_text):
        """Add rating buttons (1-5) after an assistant response."""
        # Create a frame for rating buttons
        rating_frame = tk.Frame(self.history)
        
        tk.Label(rating_frame, text="Rate: ", font=("Consolas", 8), foreground="#7F8C8D").pack(side="left")
        
        for i in range(1, 6):
            btn = tk.Button(rating_frame, text=str(i), width=2, 
                          command=lambda r=i: self._on_rate_response(r),
                          font=("Consolas", 8),
                          bg="#ECF0F1" if i <= 2 else ("#F39C12" if i == 3 else "#27AE60"),
                          fg="white" if i >= 4 else "black")
            btn.pack(side="left", padx=1)
        
        tk.Label(rating_frame, text="  ", font=("Consolas", 8)).pack(side="left")  # Spacer
        
        # Embed the frame in the text widget
        self.history.window_create("end", window=rating_frame)
        self.history.insert("end", "\n\n")  # Add spacing after rating buttons

    def _on_rate_response(self, rating):
        """Handle rating button click for the last assistant response."""
        if self._last_assistant_response is None:
            messagebox.showwarning("No response", "No response to rate.")
            return
        
        prompt, response, model_name = self._last_assistant_response
        
        try:
            entry = self.rating_collector.collect(
                prompt=prompt,
                response=response,
                model_name=model_name,
                rating=rating,
                metadata={"timestamp": time.time()}
            )
            self._append_history("system", f"[Rated {rating}/5 for '{model_name}']")
            self._log_unified("system", f"Feedback collected: {rating}/5 for {model_name}")
        except Exception as e:
            self._append_history("error", f"[Failed to record rating: {e}]")
            self._log_unified("error", f"Failed to record rating: {e}")

    def _log_train(self, message, is_error=False):
        """Append a timestamped line to the training monitor."""
        ts = time.strftime("%H:%M:%S")
        self.train_log.configure(state="normal")
        self.train_log.insert("end", f"[{ts}] ", "ts")
        self.train_log.insert("end", message + "\n", "err" if is_error else "")
        self.train_log.see("end")
        self.train_log.configure(state="disabled")
        
        # Also log to unified log viewer
        self._log_unified("error" if is_error else "training", message)

    def _log_unified(self, log_type, message):
        """Add entry to unified log viewer."""
        ts = time.strftime("%H:%M:%S")
        entry = (ts, log_type, message)
        self._all_log_entries.append(entry)
        
        # Keep only last 10000 entries to prevent memory issues
        if len(self._all_log_entries) > 10000:
            self._all_log_entries = self._all_log_entries[-10000:]
        
        # Apply current filters and display
        self._apply_log_filters()

    def _apply_log_filters(self, event=None):
        """Apply filters to unified log display."""
        if not hasattr(self, 'unified_log'):
            return
            
        search_text = self.log_search_var.get().lower() if hasattr(self, 'log_search_var') else ""
        
        # Determine which log types to show
        show_types = set()
        if hasattr(self, 'log_filter_vars'):
            for key, var in self.log_filter_vars.items():
                if var.get():
                    show_types.add(key)
        else:
            show_types = {"training", "generation", "system", "error"}
        
        self.unified_log.configure(state="normal")
        self.unified_log.delete("1.0", "end")
        
        for ts, log_type, message in self._all_log_entries:
            if log_type not in show_types:
                continue
            if search_text and search_text not in message.lower() and search_text not in ts.lower():
                continue
            
            self.unified_log.insert("end", f"[{ts}] ", "timestamp")
            self.unified_log.insert("end", f"[{log_type.upper()}] ", log_type)
            self.unified_log.insert("end", message + "\n", log_type)
        
        if self.log_autoscroll_var.get():
            self.unified_log.see("end")
        self.unified_log.configure(state="disabled")

    def _clear_unified_logs(self):
        """Clear all unified log entries."""
        self._all_log_entries.clear()
        self.unified_log.configure(state="normal")
        self.unified_log.delete("1.0", "end")
        self.unified_log.configure(state="disabled")

    def _export_logs(self):
        """Export unified logs to a text file."""
        filepath = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
            title="Export Logs"
        )
        if not filepath:
            return
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                for ts, log_type, message in self._all_log_entries:
                    f.write(f"[{ts}] [{log_type.upper()}] {message}\n")
            self._log_unified("system", f"Logs exported to {filepath}")
        except Exception as e:
            self._log_unified("error", f"Failed to export logs: {e}")

    def _set_busy(self, busy, message=""):
        self.busy = busy
        state = "disabled" if busy else "normal"
        self.train_btn.configure(state=state)
        self.train_router_btn.configure(state=state)
        self.send_btn.configure(state=state)
        self.prompt_var.configure(state=state)
        self.clear_btn.configure(state=state)
        self.create_btn.configure(state=state)
        self.delete_btn.configure(state=state)
        self.status.configure(text=message or ("Working..." if busy else "Ready"))

    # ---- background work ---------------------------------------------------

    def _run_in_background(self, label, func, *args):
        self._set_busy(True, f"{label}...")

        def worker():
            try:
                result = func(*args)
                self.queue.put(("ok", label, result))
            except Exception as exc:
                self.queue.put(("error", label, exc))

        threading.Thread(target=worker, daemon=True).start()

    def _poll_queue(self):
        try:
            while True:
                kind, label, payload = self.queue.get_nowait()
                if kind == "progress":
                    self._handle_progress(label, payload)
                elif kind == "log":
                    # Direct log to unified log (runs on main thread)
                    self._log_unified(label, payload)
                elif kind == "stream_token":
                    # Streaming token update (runs on main thread).
                    # Posted as ("stream_token", token_str, (idx, total, tps)):
                    # the token travels in the label slot, stats in the payload
                    token_str = label
                    token_idx, total_tokens, tps = payload
                    self._handle_stream_token(token_str, token_idx, total_tokens, tps)
                else:
                    self._handle_result(kind, label, payload)
        except queue.Empty:
            pass
        self.root.after(100, self._poll_queue)

    def _handle_progress(self, label, payload):
        """Update the training progress bar (runs on the main thread)."""
        if label != "Training":
            return
        done, total, loss = payload
        self.train_progress.configure(maximum=max(total, 1), value=done)
        self.progress_label.configure(text=f"Step {done}/{total}  loss {loss:.3f}")

    def _handle_stream_token(self, token_str, token_idx, total_tokens, tps):
        """Handle streaming token from generation (runs on main thread).

        Appends to the end of the history only — rewriting the whole widget
        would strip the role-tag styling from earlier messages.
        """
        self.history.configure(state="normal")
        if not self._stream_active:
            # Open the assistant block once: "AI: " label + streamed body
            if self.history.get("end-2c", "end-1c") != "\n":
                self.history.insert("end", "\n")
            self.history.insert("end", "AI: ", "assistant")
            self._stream_active = True
        self.history.insert("end", token_str)
        self.history.see("end")
        self.history.configure(state="disabled")

        # Update status bar with tokens/sec
        self.status.configure(text=f"Generating... {token_idx}/{total_tokens} tokens ({tps:.1f} tok/s)")

    # ---- actions -----------------------------------------------------------

    def on_train(self):
        if self.busy:
            return
        prompt = self.train_text.get("1.0", "end-1c").strip()
        if not prompt or prompt == "Enter training text here...":
            messagebox.showwarning("No input", "Enter some training text in the Training Data area.")
            return
        try:
            steps = int(self.steps_var.get())
            if steps < 1:
                raise ValueError
        except ValueError:
            messagebox.showwarning("Invalid input", "Training steps must be a positive whole number.")
            return
        self._log_train(f"Training started: {len(prompt)} chars, {steps} steps, model={self.model_var.get()}")
        self.train_progress.configure(maximum=steps, value=0)
        self.progress_label.configure(text=f"Step 0/{steps}")
        self._run_in_background("Training", self._train_job, prompt, steps)

    def _train_job(self, prompt, steps):
        # Use config for tokenizer and training settings
        tokenizer_cfg = self.cfg["tokenizer"]
        training_cfg = self.cfg["training"]
        
        self.tok.extend(prompt, vocab_size=tokenizer_cfg["vocab_size"], min_frequency=tokenizer_cfg["min_frequency"])
        self.tok.save(TOKENIZER_PATH)
        sync_vocab(self.model, len(self.tok.vocab))

        ids = self.tok.encode(prompt)
        if len(ids) <= self.model.ctx:
            raise ValueError(
                f"The input has {len(ids)} tokens, but training needs more than {self.model.ctx}."
            )

        # Progress callback: thread-safe, only puts messages on the queue
        def on_progress(done, total, loss):
            self.queue.put(("progress", "Training", (done, total, loss)))

        train_parallel(self.model, ids, steps=steps, lr=training_cfg["lr"], 
                       batch_size=training_cfg["batch_size"], n_negatives=training_cfg["n_negatives"],
                       workers=training_cfg["workers"], progress_cb=on_progress)
        return prompt, steps

    def on_train_router(self):
        """Train the Router model on collected routing data."""
        if self.busy:
            return
        if not os.path.exists("router_training_data.jsonl"):
            messagebox.showwarning("No data", "No router training data found. Use the chat to generate routing data first.")
            return
        try:
            steps = int(self.steps_var.get())
            if steps < 1:
                raise ValueError
        except ValueError:
            messagebox.showwarning("Invalid input", "Training steps must be a positive whole number.")
            return
        self._log_train(f"Router training started: {steps} steps")
        self.train_progress.configure(maximum=steps, value=0)
        self.progress_label.configure(text=f"Step 0/{steps}")
        self._run_in_background("Training Router", self._train_router_job, steps)

    def _train_router_job(self, steps):
        """Background job to train the Router."""
        from ai import train_router, generate_router_training_data
        from tokenizer import BPETokenizer
        
        # Load training data
        examples = []
        with open("router_training_data.jsonl", "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                examples.append(RouterTrainingExample(
                    prompt=data["prompt"],
                    subtasks=data["subtasks"]
                ))
        
        if not examples:
            self.queue.put(("error", "Router Training", "No valid training examples"))
            return
        
        # Progress callback
        def on_progress(done, total, loss, val_loss=None, lr=None):
            self.queue.put(("progress", "Router Training", (done, total, loss)))
        
        training_cfg = self.cfg["training"]
        
        train_router(self.router, self.tok, examples, 
                    steps=steps, lr=training_cfg["lr"], 
                    batch_size=training_cfg["batch_size"], 
                    n_negatives=training_cfg["n_negatives"],
                    progress_cb=on_progress)
        
        # Save the trained router
        save_model(self.router, "router.tlm")
        return steps

    def on_send_prompt(self):
        if self.busy:
            return
        prompt = self.prompt_var.get().strip()
        if not prompt:
            return
        # Clear prompt entry
        self.prompt_var.delete(0, "end")
        # Show user message immediately
        self._append_history("user", prompt)
        # Get generation settings
        try:
            n_tokens = int(self.n_tokens_var.get())
            temperature = float(self.temp_var.get())
            top_k = int(self.top_k_var.get())
            top_p = float(self.top_p_var.get())
            repetition_penalty = float(self.rep_penalty_var.get())
        except ValueError:
            n_tokens, temperature, top_k, top_p, repetition_penalty = 60, 0.8, 0, 0.0, 1.0
        self._stream_active = False  # streamed tokens open a fresh "AI:" block
        self._run_in_background("Generating", self._generate_job, prompt, n_tokens, temperature, top_k, top_p, repetition_penalty)

    def _generate_job(self, prompt, n_tokens, temperature, top_k, top_p, repetition_penalty):
        # Route the prompt using Router
        model, model_name = self._route_prompt(prompt)
        
        # Use streaming generation with callback for live updates
        def stream_callback(token_str, token_idx, total_tokens, tps):
            self.queue.put(("stream_token", token_str, (token_idx, total_tokens, tps)))
        
        # Collect routing data for Router training
        example = self._collect_routing_data(prompt, model_name, "")
        
        text = generate(model, self.tok, prompt, n_tokens=n_tokens, temperature=temperature, 
                        top_k=top_k, top_p=top_p, repetition_penalty=repetition_penalty,
                        stream_callback=stream_callback)
        
        # Collect response for routing data
        if example:
            example.subtasks[0]["response"] = text
            # Store for Router training
            self._store_routing_example(example)
        
        # Log to unified log
        self.queue.put(("log", "generation", f"Prompt: {prompt[:100]}{'...' if len(prompt) > 100 else ''}"))
        self.queue.put(("log", "generation", f"Routed to: {model_name}"))
        self.queue.put(("log", "generation", f"Response: {text[:200]}{'...' if len(text) > 200 else ''}"))
        return prompt, text, model_name

    def _store_routing_example(self, example):
        """Store routing example for Router training."""
        try:
            # Save to a file for later Router training
            import json
            with open("router_training_data.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "prompt": example.prompt,
                    "subtasks": example.subtasks
                }) + "\n")
        except Exception:
            pass

    def _route_prompt(self, prompt):
        """Route prompt using the Router model (runs on a worker thread)."""
        # Snapshot the selected model on the main thread — Tk vars must not
        # be read from background threads.
        current_name = getattr(self, "_model_name", "TinyLM")
        try:
            routes = self.router.route_multi(prompt, self.tok, MODEL_CATALOG, max_subtasks=3)
            if not routes:
                # Fallback to default model
                return self.model, current_name
            
            # Use the top route
            best_route = routes[0]
            model_name = best_route["model"]
            
            # Load the model if different from current
            if model_name != current_name:
                loader = ModelLoader(MODEL_CATALOG, tokenizer_manager=TOKENIZER_MANAGER)
                model = loader.load_model(model_name)
                return model, model_name
            return self.model, current_name
        except Exception as e:
            # Log via the queue (thread-safe) instead of touching widgets
            self.queue.put(("log", "error", f"Routing failed: {e}, falling back to current model"))
            return self.model, current_name

    def _collect_routing_data(self, prompt, model_name, response):
        """Collect routing training data from user interaction."""
        # This creates training examples for the Router
        # The Router learns to predict which model should handle which prompt
        try:
            example = RouterTrainingExample(
                prompt=prompt,
                subtasks=[{"model": model_name, "prompt": prompt}]
            )
            return example
        except Exception:
            return None

    def on_clear_history(self):
        self.history.configure(state="normal")
        self.history.delete("1.0", "end")
        self.history.configure(state="disabled")

    def on_generate_preview(self):
        """Generate with detailed preview showing tokens/sec."""
        if self.busy:
            return
        prompt = self.prompt_var.get().strip()
        if not prompt:
            return
        
        # Get generation settings
        try:
            n_tokens = int(self.n_tokens_var.get())
            temperature = float(self.temp_var.get())
            top_k = int(self.top_k_var.get())
            top_p = float(self.top_p_var.get())
            repetition_penalty = float(self.rep_penalty_var.get())
        except ValueError:
            n_tokens, temperature, top_k, top_p, repetition_penalty = 60, 0.8, 0, 0.0, 1.0
        
        # Clear prompt entry
        self.prompt_var.delete(0, "end")
        
        # Show user message
        self._append_history("user", prompt)
        self._stream_active = False

        # Run in background with preview
        self._run_in_background("Generating (Preview)", self._generate_preview_job, 
                               prompt, n_tokens, temperature, top_k, top_p, repetition_penalty)

    def _generate_preview_job(self, prompt, n_tokens, temperature, top_k, top_p, repetition_penalty):
        """Background job for generate_with_preview."""
        from ai import generate_with_preview
        result = generate_with_preview(self.model, self.tok, prompt, 
                                      n_tokens=n_tokens, temperature=temperature,
                                      top_k=top_k, top_p=top_p, 
                                      repetition_penalty=repetition_penalty)
        return prompt, result

    # ---- admin actions -----------------------------------------------------

    def refresh_catalog_list(self):
        """Refresh the Admin tab listbox from the model catalog."""
        self.admin_list.delete(0, "end")
        self._catalog_names = []
        for spec in MODEL_CATALOG.list():
            domains = ", ".join(spec.domain_keywords) or "-"
            self.admin_list.insert("end", f"{spec.name}  |  {spec.param_count:,} params  |  {domains}")
            self._catalog_names.append(spec.name)

    def on_create_model(self):
        if self.busy:
            return
        name = self.create_name_var.get().strip()
        if not name:
            messagebox.showwarning("No name", "Enter a model name (e.g. tiny-code).")
            return
        if MODEL_CATALOG.get(name) is not None:
            messagebox.showwarning("Exists", f"Model '{name}' already exists in the catalog.")
            return
        domains_raw = self.create_domains_var.get().strip()
        domains = [d.strip() for d in domains_raw.split(",") if d.strip()] or [name]
        # Use current config.yaml model section as base
        base_config = {
            "ctx": self.cfg["model"]["ctx"],
            "emb": self.cfg["model"]["emb"],
            "hidden": self.cfg["model"]["hidden"],
        }
        self._append_history("system", f"[Creating model '{name}' (domains: {', '.join(domains)})...]")
        self._run_in_background("Creating model", self._create_model_job, name, domains, base_config)

    def _create_model_job(self, name, domains, base_config):
        creator = ModelCreator(MODEL_CATALOG, "models", tokenizer_manager=TOKENIZER_MANAGER)
        return creator.create_model(name, domains, base_config=base_config)

    def on_delete_model(self):
        if self.busy:
            return
        selection = self.admin_list.curselection()
        if not selection:
            messagebox.showwarning("No selection", "Select a model in the catalog list first.")
            return
        name = self._catalog_names[selection[0]]
        if not messagebox.askyesno("Confirm delete", f"Delete model '{name}' from the catalog?\n(The .tlm file is removed too.)"):
            return
        self._run_in_background("Deleting model", self._delete_model_job, name)

    def _delete_model_job(self, name):
        spec = MODEL_CATALOG.get(name)
        removed = MODEL_CATALOG.remove(name)
        if spec is not None and os.path.exists(spec.path):
            try:
                os.remove(spec.path)
            except OSError:
                pass
        return name, removed

    # ---- settings actions --------------------------------------------------

    def on_save_config(self):
        """Validate Settings tab entries and save to config.yaml."""
        new_data = self.cfg.to_dict()
        for (section, key), (entry, cast) in self._settings_entries.items():
            raw = entry.get().strip()
            try:
                value = cast(raw)
            except ValueError:
                messagebox.showwarning(
                    "Invalid value", f"{section}.{key}: '{raw}' is not a valid {cast.__name__}."
                )
                return
            # Validate integer fields: most must be positive; some allow zero
            ZERO_OK = {"top_k", "warmup_steps", "eval_every", "checkpoint_every"}
            if isinstance(value, int) and not isinstance(value, bool):
                if key in ZERO_OK:
                    if value < 0:
                        messagebox.showwarning("Invalid value", f"{section}.{key} must be >= 0.")
                        return
                elif value < 1:
                    messagebox.showwarning("Invalid value", f"{section}.{key} must be a positive whole number.")
                    return
            new_data[section][key] = value
        gen = new_data["generation"]
        if gen["temperature"] <= 0:
            messagebox.showwarning("Invalid value", "generation.temperature must be greater than 0.")
            return
        if gen["top_k"] < 0:
            messagebox.showwarning("Invalid value", "generation.top_k must be >= 0.")
            return
        if not (0.0 <= gen["top_p"] <= 1.0):
            messagebox.showwarning("Invalid value", "generation.top_p must be between 0.0 and 1.0.")
            return
        if gen["repetition_penalty"] < 1.0:
            messagebox.showwarning("Invalid value", "generation.repetition_penalty must be >= 1.0.")
            return
        tr = new_data["training"]
        if tr["lr_scheduler"] not in ("linear", "cosine_warmup", "constant"):
            messagebox.showwarning("Invalid value",
                                   "training.lr_scheduler must be linear, cosine_warmup or constant.")
            return
        if tr["min_lr"] < 0:
            messagebox.showwarning("Invalid value", "training.min_lr must be >= 0.")
            return
        if tr["grad_clip"] < 0:
            messagebox.showwarning("Invalid value", "training.grad_clip must be >= 0.")
            return
        if not (0.0 <= tr["val_split"] < 1.0):
            messagebox.showwarning("Invalid value", "training.val_split must be in [0.0, 1.0).")
            return
        if tr["min_delta"] < 0:
            messagebox.showwarning("Invalid value", "training.min_delta must be >= 0.")
            return
        if not isinstance(tr["early_stopping"], bool):
            messagebox.showwarning("Invalid value", "training.early_stopping must be true or false.")
            return
        if not (0.0 < tr["auto_factor"] < 1.0):
            messagebox.showwarning("Invalid value", "training.auto_factor must be between 0.0 and 1.0.")
            return
        if not isinstance(tr["auto_adjust"], bool) or not isinstance(tr["auto_clip_relax"], bool):
            messagebox.showwarning("Invalid value", "training.auto_adjust and auto_clip_relax must be true or false.")
            return

        try:
            _write_config_yaml(CONFIG_PATH, new_data)
            self.cfg = Config.from_dict(new_data)
        except Exception as exc:
            messagebox.showerror("Error", f"Failed to save config:\n{exc}")
            return

        # Sync quick-access entries with new defaults
        self.steps_var.delete(0, "end")
        self.steps_var.insert(0, str(new_data["training"]["steps"]))
        self.n_tokens_var.delete(0, "end")
        self.n_tokens_var.insert(0, str(new_data["generation"]["n_tokens"]))
        self.temp_var.delete(0, "end")
        self.temp_var.insert(0, str(new_data["generation"]["temperature"]))
        self.top_k_var.delete(0, "end")
        self.top_k_var.insert(0, str(new_data["generation"]["top_k"]))
        self.top_p_var.delete(0, "end")
        self.top_p_var.insert(0, str(new_data["generation"]["top_p"]))
        self.rep_penalty_var.delete(0, "end")
        self.rep_penalty_var.insert(0, str(new_data["generation"]["repetition_penalty"]))

        self._append_history("system", f"[Config saved to {CONFIG_PATH}]")

    # ---- tokenizer manager actions -------------------------------------------

    def refresh_tokenizer_patterns(self):
        """Refresh the tokenizer patterns display."""
        try:
            patterns = self.tok.learned_patterns(min_len=2)
            # Show first 100 patterns to avoid UI overload
            display_patterns = patterns[:100]
            self.tok_patterns_text.configure(state="normal")
            self.tok_patterns_text.delete("1.0", "end")
            if display_patterns:
                self.tok_patterns_text.insert("1.0", "\n".join(display_patterns))
                if len(patterns) > 100:
                    self.tok_patterns_text.insert("end", f"\n... ({len(patterns)} total patterns)")
            else:
                self.tok_patterns_text.insert("1.0", "(no patterns learned yet)")
            self.tok_patterns_text.configure(state="disabled")
            # Update vocab/merges labels
            self.tok_vocab_label.configure(text=str(len(self.tok.vocab)))
            self.tok_merges_label.configure(text=str(len(self.tok.merges)))
            self.vocab_label.configure(text=f"Vocab: {len(self.tok.vocab)}")
        except Exception as exc:
            self.tok_patterns_text.configure(state="normal")
            self.tok_patterns_text.delete("1.0", "end")
            self.tok_patterns_text.insert("1.0", f"Error: {exc}")
            self.tok_patterns_text.configure(state="disabled")

    def on_extend_vocab(self):
        """Extend tokenizer vocabulary with new text."""
        if self.busy:
            return
        text = self.tok_extend_text.get("1.0", "end-1c").strip()
        if not text:
            messagebox.showwarning("No input", "Enter some text to extend the vocabulary.")
            return
        try:
            target_vocab = int(self.tok_target_vocab_var.get())
            min_freq = int(self.tok_min_freq_var.get())
            if target_vocab < 256:
                raise ValueError("Target vocab size must be at least 256")
            if min_freq < 1:
                raise ValueError("Min frequency must be at least 1")
        except ValueError as e:
            messagebox.showwarning("Invalid input", str(e))
            return

        self._log_train(f"Extending vocab: +{len(text)} chars, target={target_vocab}, min_freq={min_freq}")
        self._run_in_background("Extending vocab", self._extend_vocab_job, text, target_vocab, min_freq)

    def _extend_vocab_job(self, text, target_vocab, min_freq):
        """Background job to extend vocabulary."""
        # Extend the shared tokenizer
        TOKENIZER_MANAGER.extend(text, vocab_size=target_vocab, min_frequency=min_freq)
        TOKENIZER_MANAGER.save(TOKENIZER_PATH)
        
        # Sync model vocabulary (grows embeddings in place)
        sync_vocab(self.model, len(self.tok.vocab))
        
        # Save updated model
        save_model(self.model, MODEL_PATH)
        
        return target_vocab, min_freq

    def _handle_result(self, kind, label, payload):
        if kind == "error":
            self._stream_active = False
            self._set_busy(False, "Error")
            self._append_history("error", f"Error: {payload}")
            if label == "Training":
                self._log_train(f"{label} failed: {payload}", is_error=True)
                self.progress_label.configure(text="Failed")
            elif label == "Extending vocab":
                self._log_train(f"Vocab extension failed: {payload}", is_error=True)
            messagebox.showerror("Error", f"{label} failed:\n{payload}")
            return

        if label == "Training":
            prompt, steps = payload
            save_model(self.model, MODEL_PATH)
            self.vocab_label.configure(text=f"Vocab: {len(self.tok.vocab)}")
            self._append_history("system", f"[Trained on {len(prompt)} chars, {steps} steps, model saved. Vocab: {len(self.tok.vocab)}]")
            self._log_train(f"Training done: {steps} steps on {len(prompt)} chars, model saved, vocab={len(self.tok.vocab)}")
            self.train_progress.configure(maximum=max(steps, 1), value=steps)
            self.progress_label.configure(text=f"Done ({steps} steps)")
            self._set_busy(False, "Training done")
            self.update_model_inspector()
        elif label == "Training Router":
            steps = payload
            self._log_train(f"Router training done: {steps} steps")
            self.train_progress.configure(maximum=max(steps, 1), value=steps)
            self.progress_label.configure(text=f"Done ({steps} steps)")
            self._set_busy(False, "Router training done")
        elif label.startswith("Loading "):
            self.model = payload
            model_name = label.replace("Loading ", "")
            self._append_history("system", f"[Loaded model: {model_name} ({len(self.model.C)} vocab, {self.model.ctx} ctx)]")
            self._set_busy(False, "Ready")
            self.update_model_inspector()
        elif label == "Generating":
            # Plain generation finished (tokens may have been streamed live)
            prompt, text, routed_model = payload
            if self._stream_active:
                # Close the streamed block and attach rating buttons
                self.history.configure(state="normal")
                self.history.insert("end", "\n\n")
                self._add_rating_buttons(text)
                self.history.see("end")
                self.history.configure(state="disabled")
                self._last_assistant_response = (prompt, text, routed_model)
                self._stream_active = False
            else:
                # Nothing streamed (e.g. empty output); show the full response
                self._append_history("assistant", text)
                self._last_assistant_response = (prompt, text, routed_model)
            self._set_busy(False, "Ready")
            self.status.configure(text="Ready")
        elif label == "Generating (Preview)":
            prompt, result = payload
            # Result already printed to console by generate_with_preview
            # Just add to history if not already there
            self._append_history("assistant", result["text"])
            self._set_busy(False, "Ready")
            self.status.configure(text=f"Ready ({result['tokens_per_sec']:.1f} tok/s)")
        elif label == "Creating model":
            spec = payload
            self._append_history("system", f"[Created model '{spec.name}' ({spec.param_count:,} params, domains: {', '.join(spec.domain_keywords)})]")
            self.refresh_catalog_list()
            self.refresh_models()
            self._set_busy(False, "Model created")
        elif label == "Deleting model":
            name, removed = payload
            if removed:
                self._append_history("system", f"[Deleted model '{name}' from catalog]")
            else:
                self._append_history("system", f"[Model '{name}' was not in catalog]")
            self.refresh_catalog_list()
            self.refresh_models()
            self._set_busy(False, "Model deleted")
        elif label == "Extending vocab":
            target_vocab, min_freq = payload
            self.vocab_label.configure(text=f"Vocab: {len(self.tok.vocab)}")
            self.tok_vocab_label.configure(text=str(len(self.tok.vocab)))
            self.tok_merges_label.configure(text=str(len(self.tok.merges)))
            self._append_history("system", f"[Extended vocab to {len(self.tok.vocab)} tokens (target={target_vocab}, min_freq={min_freq}), model saved]")
            self._log_train(f"Vocab extended to {len(self.tok.vocab)} tokens, model saved")
            self._set_busy(False, "Vocab extended")
            self.refresh_tokenizer_patterns()
            self.update_model_inspector()

    def update_model_inspector(self):
        """Update the model inspector with current model details."""
        if self.model is None:
            self.inspector_model_name.configure(text="None")
            self.inspector_arch.configure(text="Unknown")
            self.inspector_params.configure(text="0")
            self.inspector_memory.configure(text="0 KB")
            self.inspector_vocab.configure(text="0")
            self.inspector_ctx.configure(text="0")
            self.inspector_emb.configure(text="0")
            self.inspector_hidden.configure(text="0")
            self.tensor_detail_btn.configure(state="disabled")
            return

        # Get model config
        config = self.model.get_config()
        arch = config.get("arch", "TinyLM")
        vocab_size = config.get("vocab_size", len(self.model.C))
        ctx = config.get("ctx", getattr(self.model, "ctx", 0))
        emb = config.get("emb", len(self.model.C[0]) if self.model.C else 0)
        hidden = config.get("hidden", len(self.model.b1) if self.model.b1 else 0)

        # Count parameters
        try:
            param_count = count_params(self.model)
        except Exception:
            param_count = 0

        # Estimate 4-bit quantized memory (4 bits per param = 0.5 bytes)
        memory_bytes = param_count * 0.5
        if memory_bytes >= 1024 * 1024:
            memory_str = f"{memory_bytes / (1024 * 1024):.2f} MB"
        else:
            memory_str = f"{memory_bytes / 1024:.1f} KB"

        # Update UI
        model_name = self.model_var.get()
        self.inspector_model_name.configure(text=model_name)
        self.inspector_arch.configure(text=arch)
        self.inspector_params.configure(text=f"{param_count:,}")
        self.inspector_memory.configure(text=memory_str)
        self.inspector_vocab.configure(text=str(vocab_size))
        self.inspector_ctx.configure(text=str(ctx))
        self.inspector_emb.configure(text=str(emb))
        self.inspector_hidden.configure(text=str(hidden))
        self.tensor_detail_btn.configure(state="normal")

    def on_show_tensor_details(self):
        """Show detailed tensor information in a popup window."""
        if self.model is None:
            return

        config = self.model.get_config()
        tensors = self.model.get_tensors()

        # Create detail window
        detail_win = tk.Toplevel(self.root)
        detail_win.title(f"Tensor Details - {self.model_var.get()}")
        detail_win.geometry("600x500")
        detail_win.transient(self.root)

        # Scrolled text for details
        text_area = scrolledtext.ScrolledText(detail_win, wrap="word", font=("Consolas", 9))
        text_area.pack(fill="both", expand=True, padx=10, pady=10)

        # Build detail text
        lines = []
        lines.append(f"Model: {self.model_var.get()}")
        lines.append(f"Architecture: {config.get('arch', 'Unknown')}")
        lines.append(f"Vocab Size: {config.get('vocab_size', 'N/A')}")
        lines.append(f"Context: {config.get('ctx', 'N/A')}")
        lines.append(f"Embedding Dim: {config.get('emb', 'N/A')}")
        lines.append(f"Hidden Dim: {config.get('hidden', 'N/A')}")
        lines.append("")

        total_params = 0
        for name, rows in tensors.items():
            n_rows = len(rows)
            n_cols = len(rows[0]) if rows else 0
            tensor_params = n_rows * n_cols
            total_params += tensor_params
            lines.append(f"  {name}: {n_rows} x {n_cols} = {tensor_params:,} params")

        lines.append("")
        lines.append(f"Total Parameters: {total_params:,}")

        text_area.insert("1.0", "\n".join(lines))
        text_area.configure(state="disabled")

        # Close button
        tk.Button(detail_win, text="Close", command=detail_win.destroy).pack(pady=(0, 10))


if __name__ == "__main__":
    # PySide6 (Qt6) is preferred; falls back to this Tkinter app when Qt
    # is unavailable. Force a toolkit: SCRATCHPADLM_UI=pyside6|tkinter
    from ui_base import run_gui
    run_gui()
