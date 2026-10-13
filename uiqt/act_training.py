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
    count_params,
    generate,
    generate_with_preview,
    insert_markers,
    load_model,
    mark_and_encode,
    render_marked,
    save_model,
    sync_vocab,
    train_parallel,
    train_router,
)
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
        self.train_progress.setMaximum(steps)
        self.train_progress.setValue(0)
        self.progress_label.setText(f"Step 0/{steps}")
        self._rollback_count = 0
        self.rollback_label.setText("Last rollback gap: —")
        self.worker.submit("Training", self._train_job, prompt, steps)
        self.set_busy(True, "Training...")

    def _train_job(self, prompt, steps):
        tokenizer_cfg = self.cfg["tokenizer"]
        training_cfg = self.cfg["training"]

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

        def on_progress(done, total, loss):
            self.worker.post("progress", "Training", (done, total, loss))

        # Rollback callback: report (step, spiked loss, run-best loss) to the
        # main thread so the training monitor logs it and the tab shows the gap.
        def on_rollback(step, loss, best):
            self.worker.post("rollback", "Training", (step, loss, best, loss - best))

        # Cancel callback checks the flag
        def cancel_check():
            return self._cancel_training

        def save_on_cancel(model, loss, step):
            # Save the model if loss improved (train() already kept best weights)
            save_model(model, MODEL_PATH)
            self.worker.post("log", "system", f"Training cancelled at step {step}. Loss improved ({loss:.3f}), model saved.")

        result = train_parallel(
            self.model, ids, steps=steps, lr=training_cfg["lr"],
            batch_size=training_cfg["batch_size"], n_negatives=training_cfg["n_negatives"],
            workers=training_cfg["workers"], progress_cb=on_progress,
            momentum=training_cfg.get("momentum", 0.9),
            grad_clip=training_cfg.get("grad_clip", 0.0),
            rollback_window=training_cfg.get("rollback_window", 2),
            rollback_callback=on_rollback,
            cancel_callback=cancel_check, save_on_cancel_callback=save_on_cancel,
        )
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

        def on_progress(done, total, loss, val_loss=None, lr=None):
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
        # Save the trained router unless the run was cancelled without improvement
        # (train() restores initial weights in that case).
        if not (isinstance(result, dict) and result.get("cancelled")):
            save_model(self.router, "router.tlm")
        return steps, result

    # ---- admin actions -------------------------------------------------------
