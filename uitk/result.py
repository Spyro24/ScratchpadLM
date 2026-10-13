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


class ResultMixin:
    def _handle_result(self, kind, label, payload):
        if kind == "error":
            self._stream_active = False
            self._set_busy(False, "Error")
            try:
                self.cancel_train_btn.configure(state="disabled")
                self.train_btn.configure(state="normal")
                self.train_router_btn.configure(state="normal")
                if label == "Training":
                    self._restore_training_text()
            except Exception:
                pass
            # Chat shows only conversation (user/assistant) plus model-selection
            # notices and generation errors. Training/admin/vocab/config errors
            # go to the Training monitor / unified Logs only.
            if label in ("Generating", "Generating (Preview)") or label.startswith("Loading "):
                self._append_history("error", f"Error: {payload}")
            if label in ("Training", "Training Router"):
                self._log_train(f"{label} failed: {payload}", is_error=True)
                self._log_unified("error", f"{label} failed: {payload}")
                self.progress_label.configure(text="Failed")
            elif label == "Extending vocab":
                self._log_train(f"Vocab extension failed: {payload}", is_error=True)
                self._log_unified("error", f"Vocab extension failed: {payload}")
            else:
                self._log_unified("error", f"{label} failed: {payload}")
            messagebox.showerror("Error", f"{label} failed:\n{payload}")
            return

        # Reset training buttons for any training completion
        if label in ("Training", "Training Router", "Generating", "Generating (Preview)"):
            self.cancel_train_btn.configure(state="disabled")
            self.train_btn.configure(state="normal")
            self.train_router_btn.configure(state="normal")
            if label == "Training":
                try:
                    self._restore_training_text()
                except Exception:
                    pass

        if label == "Training":
            if isinstance(payload, tuple) and len(payload) == 3:
                prompt, steps, result = payload
            else:
                prompt, steps = payload
                result = None
            cancelled = isinstance(result, dict) and result.get("cancelled")
            if cancelled:
                info = result or {}
                self._log_train(f"Training cancelled after {info.get('steps_done', steps)} steps "
                                f"(initial_loss={info.get('initial_loss')}, best_loss={info.get('best_loss')}, "
                                f"rollbacks={info.get('rollbacks', 0)})")
                self._log_unified("system", "Training cancelled by user.")
                self.train_progress.configure(maximum=max(steps, 1), value=info.get("steps_done", 0))
                self.progress_label.configure(text=f"Cancelled ({info.get('steps_done', 0)} steps)")
                self._set_busy(False, "Training cancelled")
            else:
                info = result if isinstance(result, dict) else {}
                # Training logs only go to training tab / unified logs, not chat
                if info.get("stats_reverted"):
                    self._log_train(f"Training done: {steps} steps, no improvement "
                                    f"(best stays {info.get('stats_best')}) - reverted, model file kept")
                    self._log_unified("system", "Training done without improvement - reverted to best weights.")
                else:
                    save_model(self.model, MODEL_PATH)
                    self._log_train(f"Training done: {steps} steps on {len(prompt)} chars, model saved, vocab={len(self.tok.vocab)}, "
                                    f"rollbacks={info.get('rollbacks', 0)}")
                    self._log_unified("system", f"Training done: {steps} steps, model saved.")
                self.vocab_label.configure(text=f"Vocab: {len(self.tok.vocab)}")
                self.train_progress.configure(maximum=max(steps, 1), value=steps)
                self.progress_label.configure(text=f"Done ({steps} steps)")
                self._set_busy(False, "Training done")
            self.update_model_inspector()
        elif label == "Training Router":
            if isinstance(payload, tuple) and len(payload) == 2:
                steps, result = payload
            else:
                steps, result = payload, None
            cancelled = isinstance(result, dict) and result.get("cancelled")
            if cancelled:
                info = result or {}
                self._log_train(f"Router training cancelled after {info.get('steps_done', steps)} steps")
                self._log_unified("system", "Router training cancelled by user.")
                self.progress_label.configure(text=f"Cancelled ({info.get('steps_done', 0)} steps)")
                self._set_busy(False, "Router training cancelled")
            else:
                self._log_train(f"Router training done: {steps} steps")
                self._log_unified("system", f"Router training done: {steps} steps")
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
            self._log_unified("system", f"Created model '{spec.name}' ({spec.param_count:,} params, domains: {', '.join(spec.domain_keywords)})")
            self.refresh_catalog_list()
            self.refresh_models()
            self._set_busy(False, "Model created")
        elif label == "Deleting model":
            name, removed = payload
            if removed:
                self._log_unified("system", f"Deleted model '{name}' from catalog")
            else:
                self._log_unified("system", f"Model '{name}' was not in catalog")
            self.refresh_catalog_list()
            self.refresh_models()
            self._set_busy(False, "Model deleted")
        elif label == "Analyzing data":
            self._handle_analyze_result(payload)
            self._set_busy(False, "Analysis done")
        elif label == "Extending vocab":
            target_vocab, min_freq = payload
            self.vocab_label.configure(text=f"Vocab: {len(self.tok.vocab)}")
            self.tok_vocab_label.configure(text=str(len(self.tok.vocab)))
            self.tok_merges_label.configure(text=str(len(self.tok.merges)))
            self._log_unified("system", f"Extended vocab to {len(self.tok.vocab)} tokens (target={target_vocab}, min_freq={min_freq}), model saved")
            self._log_train(f"Vocab extended to {len(self.tok.vocab)} tokens, model saved")
            self._set_busy(False, "Vocab extended")
            self.refresh_tokenizer_patterns()
            self.update_model_inspector()
