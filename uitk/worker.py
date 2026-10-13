"""Tkinter UI mixin (stdlib fallback). Part of the `uitk/` package split from `ui.py`."""
import json
import math
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


class WorkerMixin:
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
                elif kind == "rollback":
                    self._handle_rollback(label, payload)
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

    def _handle_rollback(self, label, payload):
        """Log a loss-spike rollback in the training monitor and show the gap
        (current loss minus smallest loss in the run) in the training tab."""
        if label != "Training":
            return
        step, loss, best, gap = payload
        count = getattr(self, "_rollback_count", 0) + 1
        self._rollback_count = count
        gap_txt = f"{gap:+.3f}" if math.isfinite(gap) else str(gap)
        self._log_train(f"Rollback at step {step}: loss {loss:.3f} > run best {best:.3f} "
                        f"(gap {gap_txt}) -> restored best weights")
        try:
            self.rollback_label.configure(
                text=f"Last rollback gap: {gap_txt} (step {step}, #{count})")
        except Exception:
            pass

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
        total = total_tokens if total_tokens is not None else "…"
        self.status.configure(text=f"Generating... {token_idx}/{total} tokens ({tps:.1f} tok/s)")

    # ---- actions -----------------------------------------------------------
