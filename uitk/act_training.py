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


class TrainingActionsMixin:
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
        self._cancel_training = False
        self.cancel_train_btn.configure(state="normal")
        self.train_btn.configure(state="disabled")
        self.train_router_btn.configure(state="disabled")
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

        # Cancel callback checks the flag
        def cancel_check():
            return self._cancel_training

        def save_on_cancel(model, loss, step):
            # Save the model if loss improved (train() already kept best weights)
            save_model(model, MODEL_PATH)
            self.queue.put(("log", "system", f"Training cancelled at step {step}. Loss improved ({loss:.3f}), model saved."))

        result = train_parallel(self.model, ids, steps=steps, lr=training_cfg["lr"],
                       batch_size=training_cfg["batch_size"], n_negatives=training_cfg["n_negatives"],
                       workers=training_cfg["workers"], progress_cb=on_progress,
                       cancel_callback=cancel_check, save_on_cancel_callback=save_on_cancel)
        return prompt, steps, result

    def on_cancel_training(self):
        """Cancel the current training job."""
        if self.busy:
            self._cancel_training = True
            self._log_train("Training cancellation requested...")
            self.cancel_train_btn.configure(state="disabled")

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
        self._cancel_training = False
        self.cancel_train_btn.configure(state="normal")
        self.train_btn.configure(state="disabled")
        self.train_router_btn.configure(state="disabled")
        self._log_train(f"Router training started: {steps} steps")
        self.train_progress.configure(maximum=steps, value=0)
        self.progress_label.configure(text=f"Step 0/{steps}")
        self._run_in_background("Training Router", self._train_router_job, steps)

    def _train_router_job(self, steps):
        """Background job to train the Router."""
        from tinylm import train_router, generate_router_training_data
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
        
        # Cancel callback checks the flag
        def cancel_check():
            return self._cancel_training
        
        training_cfg = self.cfg["training"]
        
        result = train_router(self.router, self.tok, examples,
                    steps=steps, lr=training_cfg["lr"],
                    batch_size=training_cfg["batch_size"],
                    n_negatives=training_cfg["n_negatives"],
                    progress_cb=on_progress,
                    cancel_callback=cancel_check)

        # Save the trained router unless the run was cancelled without improvement
        # (train() restores initial weights in that case).
        if not (isinstance(result, dict) and result.get("cancelled")):
            save_model(self.router, "router.tlm")
        return steps, result
