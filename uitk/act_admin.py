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


class AdminActionsMixin:
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
        self._log_unified("system", f"Creating model '{name}' (domains: {', '.join(domains)})...")
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

    def on_analyze_data(self):
        """Analyze training data and suggest model name."""
        text = self.analyze_data_var.get().strip()
        if not text or text == "Paste training text here to analyze...":
            messagebox.showwarning("No input", "Paste some training text to analyze.")
            return
        
        # Get existing model names
        existing_names = {spec.name for spec in MODEL_CATALOG.list()}
        
        # Run analysis in background
        self._run_in_background("Analyzing data", self._analyze_data_job, text, existing_names)

    def _analyze_data_job(self, text, existing_names):
        """Background job to analyze training data."""
        from tokenizer import analyze_training_data
        result = analyze_training_data(text, existing_names)
        return result

    def _handle_analyze_result(self, result):
        """Handle the analysis result."""
        domains_str = ", ".join(result["domains"]) if result["domains"] else "none detected"
        msg = (f"Domains: {domains_str} | "
               f"Suggested: {result['suggested_name']} | "
               f"Confidence: {result['confidence']:.1%} | "
               f"Action: {result['train_or_create']}")
        self.analyze_result_var.configure(text=msg)
        # Auto-fill the name and domains
        self.create_name_var.delete(0, "end")
        self.create_name_var.insert(0, result["suggested_name"])
        self.create_domains_var.delete(0, "end")
        self.create_domains_var.insert(0, ", ".join(result["domains"]))
