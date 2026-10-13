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


class CoreMixin:
    def __init__(self, root):
        self.root = root
        root.title("ScratchpadLM (TinyLM)")
        root.geometry("900x720")

        self.queue = queue.Queue()
        self.busy = False
        self._cancel_training = False  # Flag to signal training cancellation
        self.cfg = cfg  # Store config reference

        # Apply theme from config
        theme = self.cfg.get("ui", "theme", "system")
        self.current_theme = theme
        apply_tk_theme(root, theme)

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
        self._log_unified("system", f"Refreshed model list: {', '.join(models)}")

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
