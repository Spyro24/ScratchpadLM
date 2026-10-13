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


class InfoMixin:
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
