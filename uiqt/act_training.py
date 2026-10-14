"""PySide6 UI mixin. Part of the `uiqt/` package split from `ui_qt.py`."""
import json
import os
import time
from pathlib import Path

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QFont, QTextCursor
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QStatusBar,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from tinylm import (
    Router,
    RouterConfig,
    RouterTrainingExample,
    TinyLM,
    TrainingStats,
    apply_stats,
    count_params,
    generate,
    generate_with_preview,
    insert_markers,
    load_best,
    load_model,
    mark_and_encode,
    render_marked,
    save_model,
    sync_vocab,
    train,
    train_parallel,
    train_router,
)
from tinylm.training import _cuda_available, _accel_available, _rwkv_available
from autotrain import RatingCollector
from config import Config
from model_mgmt import MODEL_CATALOG, ModelCreator, ModelLoader
from registry import MODEL_REGISTRY, TOKENIZER_MANAGER
from ui_base import (
    CONFIG_PATH,
    MODEL_PATH,
    PALETTE,
    PALETTE_LIGHT,
    PALETTE_DARK,
    TOKENIZER_PATH,
    BackgroundWorker,
    get_available_models,
    load_config,
    parse_bool,
    qss_theme,
    get_palette,
    write_config_yaml,
)
from uiqt.widgets import MONO, MessageBubble, AssistantBubble


class TrainingActionsMixin:
    def on_cancel_training(self):
        """Cancel the current training job."""
        if self.busy:
            self._cancel_training = True
            self._log_train("Training cancellation requested...")
            self.cancel_train_btn.setEnabled(False)

    def on_train(self):
        if self.busy:
            return
        prompt = self.train_text.toPlainText().strip()
        if not prompt or prompt == "Enter training text here...":
            self.show_warning("No input", "Enter some training text in the Training Data area.")
            return
        steps = self.steps_spin.value()
        self._cancel_training = False
        self.cancel_train_btn.setEnabled(True)
        self.train_btn.setEnabled(False)
        self.train_router_btn.setEnabled(False)
        self._log_train(f"Training started: {len(prompt)} chars, {steps} steps, model={self._model_name}")
        prev_best = TrainingStats().best_loss(self._model_name)
        if prev_best is not None:
            self._log_train(f"Previous best loss for '{self._model_name}': {prev_best:.4f}")
        self.train_progress.setMaximum(steps)
        self.train_progress.setValue(0)
        self.progress_label.setText(f"Step 0/{steps}")
        self._rollback_count = 0
        self.rollback_label.setText("Last rollback gap: —")
        # Show the marked-up training text and lock the box while training
        # runs; the raw text is restored on completion.
        self._train_text_raw = prompt
        self.train_text.setPlainText(insert_markers(prompt))
        self.train_text.setReadOnly(True)
        
        # Determine which kernel will be used (before training starts)
        training_cfg = self.cfg["training"]
        use_c = training_cfg.get("use_c", None)
        advanced_features = (
            False or  # val_ids
            0.0 > 0 or  # val_split
            0 > 0 or  # eval_every
            False or  # early_stopping
            0 > 0 or  # checkpoint_every
            False or  # cancel_callback
            False  # optimizer in ("apollo", "apollo_mini") - we use train_parallel
        )
        if use_c is None:
            use_c = _cuda_available() and not advanced_features
        if use_c == "cuda":
            kernel_name = "CUDA" if _cuda_available() and not advanced_features else "Python (CUDA fallback)"
        elif use_c == "c":
            kernel_name = "C" if _accel_available() and not advanced_features else "Python (C fallback)"
        elif use_c:
            if _cuda_available() and not advanced_features:
                kernel_name = "CUDA (auto)"
            elif _accel_available() and not advanced_features:
                kernel_name = "C (auto)"
            else:
                kernel_name = "Python (auto)"
        else:
            kernel_name = "Python"
        
        self.kernel_indicator.setText(f"Kernel: {kernel_name}")
        
        self.worker.submit("Training", self._train_job, prompt, steps,
                           self._model_name, self.auto_adjust_cb.isChecked())
        self.set_busy(True, "Training...")

    def _restore_training_text(self):
        """Restore the raw training text and unlock the box."""
        raw = getattr(self, "_train_text_raw", None)
        if raw is not None:
            self.train_text.setPlainText(raw)
            del self._train_text_raw
        self.train_text.setReadOnly(False)

    def _train_job(self, prompt, steps, model_name, auto_adjust):
        tokenizer_cfg = self.cfg["tokenizer"]
        training_cfg = self.cfg["training"]

        # Load the persisted best weights BEFORE extending/syncing: afterwards
        # the model is bigger than the snapshot and the counts won't match.
        if load_best(model_name, self.model):
            self.worker.post("log", "Training",
                             f"Resuming '{model_name}' from best weights.")

        self.tok.extend(
            prompt,
            vocab_size=tokenizer_cfg["vocab_size"],
            min_frequency=tokenizer_cfg["min_frequency"],
        )
        self.tok.save(TOKENIZER_PATH)
        sync_vocab(self.model, len(self.tok.vocab))

        marked = insert_markers(prompt)
        sample = marked[:300]
        self.worker.post("log", "Training",
                         f"Training text with markers: {render_marked(sample + ('...' if len(marked) > 300 else ''), color=False)}")
        ids = mark_and_encode(prompt, self.tok, len(self.tok.vocab))
        if len(ids) <= self.model.ctx:
            raise ValueError(
                f"The input has {len(ids)} tokens, but training needs more than {self.model.ctx}."
            )

        # Start from the persisted best weights so training continues from
        # the best known point instead of drifted weights.
        if load_best(model_name, self.model):
            self.worker.post("log", "Training",
                             f"Resuming '{model_name}' from best weights.")

        def on_progress(done, total, loss, val_loss=None, lr=None, **aa):
            self.worker.post("progress", "Training", (done, total, loss, val_loss, lr,
                             aa.get("aa_mult"), aa.get("aa_momentum"), aa.get("aa_grad_clip"),
                             aa.get("aa_reduce"), aa.get("aa_bad"), aa.get("aa_best")))

        # Rollback callback: report (step, spiked loss, run-best loss) to the
        # main thread so the training monitor logs it and the tab shows the gap.
        def on_rollback(step, loss, best):
            self.worker.post("rollback", "Training", (step, loss, best, loss - best))

        # Cancel callback checks the flag
        def cancel_check():
            return self._cancel_training

        def save_on_cancel(model, loss, step):
            # Save only when the run also beats the stored best, so a
            # cancelled run can never regress the model file.
            prev = TrainingStats().best_loss(model_name)
            if prev is not None and loss >= prev:
                self.worker.post("log", "system", f"Training cancelled at step {step}. No improvement over best ({prev:.4f}), model file kept.")
                return
            save_model(model, MODEL_PATH)
            self.worker.post("log", "system", f"Training cancelled at step {step}. Loss improved ({loss:.3f}), model saved.")

        is_rwkv = type(self.model).__name__ == "RWKVModel"
        if is_rwkv:
            # RWKV trains single-process via train(): GPU WKV kernel when the
            # RWKV lib is built and the optimizer is SGD (CPU fallback stays).
            # The parallel/C kernels remain legacy-MLP only.
            self.worker.post("log", "Training",
                             "RWKVModel detected: using single-process train() "
                             "(GPU WKV kernel when available, else CPU).")
        train_fn = train if is_rwkv else train_parallel
        batch_size = self.batch_size_spin.value()
        train_kw = dict(
            lr=training_cfg["lr"],
            batch_size=batch_size, n_negatives=training_cfg["n_negatives"],
            progress_cb=on_progress,
            lr_scheduler=training_cfg.get("lr_scheduler", "cosine_warmup"),
            warmup_steps=training_cfg.get("warmup_steps", 100),
            min_lr=training_cfg.get("min_lr", 0.0),
            momentum=training_cfg.get("momentum", 0.9),
            grad_clip=training_cfg.get("grad_clip", 0.0),
            rollback_window=training_cfg.get("rollback_window", 2),
            rollback_callback=on_rollback,
            auto_adjust=auto_adjust,
            auto_factor=training_cfg.get("auto_factor", 0.5),
            auto_patience=training_cfg.get("auto_patience", 10),
            auto_clip_relax=training_cfg.get("auto_clip_relax", True),
            # New AutoAdjust features
            use_cosine_restarts=training_cfg.get("use_cosine_restarts", False),
            restart_period=training_cfg.get("restart_period", 0),
            cyclic_lr=training_cfg.get("cyclic_lr", False),
            cycle_length=training_cfg.get("cycle_length", 0),
            use_frequency_neg_sampling=training_cfg.get("use_frequency_neg_sampling", False),
            n_negatives_start=training_cfg.get("n_negatives_start"),
            n_negatives_end=training_cfg.get("n_negatives_end", 64),
            n_negatives_decay=training_cfg.get("n_negatives_decay", "exponential"),
            optimizer=training_cfg.get("optimizer", "sgd"),
            cancel_callback=cancel_check, save_on_cancel_callback=save_on_cancel,
        )
        if not is_rwkv:
            train_kw["workers"] = training_cfg["workers"]
            # Batched compute + GPU routing are parallel-path only.
            train_kw["batch_steps"] = self.batch_steps_spin.value()
            train_kw["use_gpu"] = self.use_gpu_cb.isChecked()
        result = train_fn(self.model, ids, steps=steps, **train_kw)
        stats_info = apply_stats(model_name, self.model, result)
        result = {**result, "stats_improved": stats_info["improved"],
                  "stats_reverted": stats_info["reverted"],
                  "stats_best": stats_info["best_loss"]}
        return prompt, steps, result

    def on_train_router(self):
        if self.busy:
            return
        if not os.path.exists("router_training_data.jsonl"):
            self.show_warning(
                "No data",
                "No router training data found. Use the chat to generate routing data first.",
            )
            return
        steps = self.steps_spin.value()
        self._cancel_training = False
        self.cancel_train_btn.setEnabled(True)
        self.train_btn.setEnabled(False)
        self.train_router_btn.setEnabled(False)
        self._log_train(f"Router training started: {steps} steps")
        self.train_progress.setMaximum(steps)
        self.train_progress.setValue(0)
        self.progress_label.setText(f"Step 0/{steps}")
        self.worker.submit("Training Router", self._train_router_job, steps)
        self.set_busy(True, "Training Router...")

    def _train_router_job(self, steps):
        examples = []
        with open("router_training_data.jsonl", "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                examples.append(RouterTrainingExample(prompt=data["prompt"], subtasks=data["subtasks"]))
        if not examples:
            raise ValueError("No valid training examples")

        # Warm start from the persisted best router weights so routing training
        # continues from the best known point.
        if load_best("router", self.router):
            self.worker.post("log", "Training Router",
                             "Resuming 'router' from best weights.")

        def on_progress(done, total, loss, val_loss=None, lr=None, **aa):
            self.worker.post("progress", "Router Training", (done, total, loss))

        # Cancel callback checks the flag
        def cancel_check():
            return self._cancel_training

        training_cfg = self.cfg["training"]
        result = train_router(
            self.router, self.tok, examples,
            steps=steps, lr=training_cfg["lr"],
            batch_size=training_cfg["batch_size"],
            n_negatives=training_cfg["n_negatives"],
            momentum=training_cfg.get("momentum", 0.9),
            progress_cb=on_progress,
            cancel_callback=cancel_check,
        )
        # Save only on improvement over the stored best; regression reverts.
        stats_info = apply_stats("router", self.router, result)
        result = {**result, "stats_improved": stats_info["improved"],
                  "stats_reverted": stats_info["reverted"],
                  "stats_best": stats_info["best_loss"]}
        if not stats_info["reverted"] and not (isinstance(result, dict) and result.get("cancelled")):
            save_model(self.router, "router.tlm")
        return steps, result

    # ---- admin actions -------------------------------------------------------
