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
    load_model,
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


class ResultMixin:
    def _handle_result(self, kind: str, label: str, payload):
        if kind == "error":
            self._stream_bubble = None
            self.set_busy(False, "Error")
            try:
                self.cancel_train_btn.setEnabled(False)
                self.train_btn.setEnabled(True)
                self.train_router_btn.setEnabled(True)
                if label == "Training":
                    self._restore_training_text()
            except Exception:
                pass
            # Chat shows only conversation plus model-selection notices and
            # generation errors. Training/admin/vocab/config errors go to
            # the Training monitor / unified Logs only.
            if label in ("Generating", "Generating (Preview)") or label.startswith("Loading "):
                self.append_history("error", f"Error: {payload}")
            if label in ("Training", "Training Router"):
                self._log_train(f"{label} failed: {payload}", is_error=True)
                self.log_unified("error", f"{label} failed: {payload}")
                self.progress_label.setText("Failed")
            elif label == "Extending vocab":
                self._log_train(f"Vocab extension failed: {payload}", is_error=True)
                self.log_unified("error", f"Vocab extension failed: {payload}")
            else:
                self.log_unified("error", f"{label} failed: {payload}")
            self.show_error(f"{label} failed", str(payload))
            return

        # Reset training buttons for any training completion
        if label in ("Training", "Training Router", "Generating", "Generating (Preview)"):
            self.cancel_train_btn.setEnabled(False)
            self.train_btn.setEnabled(True)
            self.train_router_btn.setEnabled(True)
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
                self.log_unified("system", "Training cancelled by user.")
                self.train_progress.setMaximum(max(steps, 1))
                self.train_progress.setValue(info.get("steps_done", 0))
                self.progress_label.setText(f"Cancelled ({info.get('steps_done', 0)} steps)")
                self.set_busy(False, "Training cancelled")
            else:
                info = result if isinstance(result, dict) else {}
                # Training logs only go to training tab / unified logs, not chat
                if info.get("stats_reverted"):
                    self._log_train(
                        f"Training done: {steps} steps, no improvement "
                        f"(best stays {info.get('stats_best')}) - reverted, model file kept"
                    )
                    self.log_unified("system", "Training done without improvement - reverted to best weights.")
                else:
                    save_model(self.model, MODEL_PATH)
                    self._log_train(
                        f"Training done: {steps} steps on {len(prompt)} chars, "
                        f"model saved, vocab={len(self.tok.vocab)}, "
                        f"rollbacks={info.get('rollbacks', 0)}"
                    )
                    self.log_unified("system", f"Training done: {steps} steps, model saved.")
                self.vocab_label.setText(f"Vocab: {len(self.tok.vocab)}")
                self.train_progress.setMaximum(max(steps, 1))
                self.train_progress.setValue(steps)
                self.progress_label.setText(f"Done ({steps} steps)")
                self.set_busy(False, "Training done")
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
                self.log_unified("system", "Router training cancelled by user.")
                self.progress_label.setText(f"Cancelled ({info.get('steps_done', 0)} steps)")
                self.set_busy(False, "Router training cancelled")
            else:
                self._log_train(f"Router training done: {steps} steps")
                self.train_progress.setMaximum(max(steps, 1))
                self.train_progress.setValue(steps)
                self.progress_label.setText(f"Done ({steps} steps)")
                self.set_busy(False, "Router training done")
        elif label.startswith("Loading "):
            self.model = payload
            model_name = label.replace("Loading ", "")
            self.append_history(
                "system",
                f"[Loaded model: {model_name} "
                f"({len(self.model.C)} vocab, {self.model.ctx} ctx)]",
            )
            self.set_busy(False, "Ready")
            self.update_model_inspector()
        elif label == "Generating":
            prompt, text, routed_model = payload
            self._last_assistant_response = (prompt, text, routed_model)
            if self._stream_bubble is not None:
                # Tokens were streamed live: close the bubble, enable feedback
                session = self._sessions[self._current_session]
                if session["messages"]:
                    session["messages"][-1].update(
                        {"prompt": prompt, "model": routed_model, "text": text}
                    )
                self._rebuild_chat_area()
                self._stream_bubble = None
            else:
                self._append_message("assistant", text, prompt=prompt, model=routed_model)
            self.set_busy(False, "Ready")
            self.status_label.setText("Ready")
        elif label == "Generating (Preview)":
            prompt, result = payload
            self._last_assistant_response = (prompt, result["text"], self._model_name)
            self._append_message(
                "assistant", result["text"], prompt=prompt, model=self._model_name
            )
            self.set_busy(False, "Ready")
            self.status_label.setText(f"Ready ({result['tokens_per_sec']:.1f} tok/s)")
        elif label == "Creating model":
            spec = payload
            self.log_unified("system", f"Created model '{spec.name}' ({spec.param_count:,} params, domains: {', '.join(spec.domain_keywords)})")
            self.refresh_catalog_list()
            self.refresh_models()
            self.set_busy(False, "Model created")
        elif label == "Deleting model":
            name, removed = payload
            if removed:
                self.log_unified("system", f"Deleted model '{name}' from catalog")
            else:
                self.log_unified("system", f"Model '{name}' was not in catalog")
            self.refresh_catalog_list()
            self.refresh_models()
            self.set_busy(False, "Model deleted")
        elif label == "Extending vocab":
            target_vocab, min_freq = payload
            self.vocab_label.setText(f"Vocab: {len(self.tok.vocab)}")
            self.tok_vocab_label.setText(str(len(self.tok.vocab)))
            self.tok_merges_label.setText(str(len(self.tok.merges)))
            self.log_unified("system", f"Extended vocab to {len(self.tok.vocab)} tokens (target={target_vocab}, min_freq={min_freq}), model saved")
            self._log_train(
                f"Vocab extended to {len(self.tok.vocab)} tokens, model saved"
            )
            self.set_busy(False, "Vocab extended")
            self.refresh_tokenizer_patterns()
            self.update_model_inspector()
        else:
            self.set_busy(False, "Ready")

    # ---- logging -------------------------------------------------------------
