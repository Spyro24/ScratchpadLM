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
from tinylm import TinyLM, train, generate, load_model, save_model, sync_vocab, train_parallel, count_params, Router, RouterTrainingExample, insert_markers, mark_and_encode, render_marked, TrainingStats, apply_stats, load_best
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
        prev_best = TrainingStats().best_loss(self.model_var.get())
        if prev_best is not None:
            self._log_train(f"Previous best loss for '{self.model_var.get()}': {prev_best:.4f}")
        self.train_progress.configure(maximum=steps, value=0)
        self.progress_label.configure(text=f"Step 0/{steps}")
        self._rollback_count = 0
        self.rollback_label.configure(text="Last rollback gap: —")
        # Show the marked-up training text (yellow markers) and lock the box
        # while training runs; the raw text is restored on completion.
        self._train_text_raw = prompt
        self._show_marked_training_text(insert_markers(prompt))
        self._run_in_background("Training", self._train_job, prompt, steps,
                                self.model_var.get(), self.auto_adjust_var.get())

    def _show_marked_training_text(self, marked):
        """Display marked training text with yellow marker tags, input locked."""
        self.train_text.configure(state="normal")
        self.train_text.delete("1.0", "end")
        self.train_text.insert("1.0", marked)
        self.train_text.tag_config("marker", foreground=PALETTE["warning"])
        self.train_text.tag_remove("marker", "1.0", "end")
        for short in ("[EOL]", "[EOS]", "[EOT]"):
            idx = "1.0"
            while True:
                idx = self.train_text.search(short, idx, stopindex="end")
                if not idx:
                    break
                self.train_text.tag_add("marker", idx, f"{idx}+{len(short)}c")
                idx = f"{idx}+{len(short)}c"
        self.train_text.configure(state="disabled")

    def _restore_training_text(self):
        """Restore the raw training text and unlock the box."""
        raw = getattr(self, "_train_text_raw", None)
        self.train_text.configure(state="normal")
        if raw is not None:
            self.train_text.delete("1.0", "end")
            self.train_text.insert("1.0", raw)
            del self._train_text_raw

    def _train_job(self, prompt, steps, model_name, auto_adjust):
        # Use config for tokenizer and training settings
        tokenizer_cfg = self.cfg["tokenizer"]
        training_cfg = self.cfg["training"]

        # Load the persisted best weights BEFORE extending/syncing: afterwards
        # the model is bigger than the snapshot and the counts won't match.
        if load_best(model_name, self.model):
            self.queue.put(("log", "Training",
                            f"Resuming '{model_name}' from best weights."))

        self.tok.extend(prompt, vocab_size=tokenizer_cfg["vocab_size"], min_frequency=tokenizer_cfg["min_frequency"])
        self.tok.save(TOKENIZER_PATH)
        sync_vocab(self.model, len(self.tok.vocab))

        marked = insert_markers(prompt)
        sample = marked[:300]
        self.queue.put(("log", "Training",
                        f"Training text with markers: {render_marked(sample + ('...' if len(marked) > 300 else ''), color=False)}"))
        ids = mark_and_encode(prompt, self.tok, len(self.tok.vocab))
        if len(ids) <= self.model.ctx:
            raise ValueError(
                f"The input has {len(ids)} tokens, but training needs more than {self.model.ctx}."
            )

        # Start from the persisted best weights so training continues from
        # the best known point instead of drifted weights.
        if load_best(model_name, self.model):
            self.queue.put(("log", "Training",
                            f"Resuming '{model_name}' from best weights."))

        # Progress callback: thread-safe, only puts messages on the queue
        def on_progress(done, total, loss):
            self.queue.put(("progress", "Training", (done, total, loss)))

        # Rollback callback: report (step, spiked loss, run-best loss) to the
        # main thread so the training monitor logs it and the tab shows the gap.
        def on_rollback(step, loss, best):
            self.queue.put(("rollback", "Training", (step, loss, best, loss - best)))

        # Cancel callback checks the flag
        def cancel_check():
            return self._cancel_training

        def save_on_cancel(model, loss, step):
            # Save only when the run also beats the stored best, so a
            # cancelled run can never regress the model file.
            prev = TrainingStats().best_loss(model_name)
            if prev is not None and loss >= prev:
                self.queue.put(("log", "system", f"Training cancelled at step {step}. No improvement over best ({prev:.4f}), model file kept."))
                return
            save_model(model, MODEL_PATH)
            self.queue.put(("log", "system", f"Training cancelled at step {step}. Loss improved ({loss:.3f}), model saved."))

        result = train_parallel(self.model, ids, steps=steps, lr=training_cfg["lr"],
                       batch_size=training_cfg["batch_size"], n_negatives=training_cfg["n_negatives"],
                       workers=training_cfg["workers"], progress_cb=on_progress,
                       momentum=training_cfg.get("momentum", 0.9),
                       grad_clip=training_cfg.get("grad_clip", 0.0),
                       rollback_window=training_cfg.get("rollback_window", 2),
                       rollback_callback=on_rollback,
                       auto_adjust=auto_adjust,
                       auto_factor=training_cfg.get("auto_factor", 0.5),
                       auto_patience=training_cfg.get("auto_patience", 10),
                       auto_clip_relax=training_cfg.get("auto_clip_relax", True),
                       cancel_callback=cancel_check, save_on_cancel_callback=save_on_cancel)
        stats_info = apply_stats(model_name, self.model, result)
        result = {**result, "stats_improved": stats_info["improved"],
                  "stats_reverted": stats_info["reverted"],
                  "stats_best": stats_info["best_loss"]}
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
        from tinylm import train_router, generate_router_training_data, apply_stats
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
                    momentum=training_cfg.get("momentum", 0.9),
                    progress_cb=on_progress,
                    cancel_callback=cancel_check)

        # Save only on improvement over the stored best; regression reverts.
        stats_info = apply_stats("router", self.router, result)
        result = {**result, "stats_improved": stats_info["improved"],
                  "stats_reverted": stats_info["reverted"],
                  "stats_best": stats_info["best_loss"]}
        if not stats_info["reverted"] and not (isinstance(result, dict) and result.get("cancelled")):
            save_model(self.router, "router.tlm")
        return steps, result
