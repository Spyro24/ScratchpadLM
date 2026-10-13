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


class WorkerMixin:
    def _poll_worker(self):
        for kind, label, payload in self.worker.drain():
            if kind == "progress":
                if label in ("Training", "Router Training"):
                    done, total, loss = payload
                    self.train_progress.setMaximum(max(total, 1))
                    self.train_progress.setValue(done)
                    self.progress_label.setText(f"Step {done}/{total}  loss {loss:.3f}")
            elif kind == "log":
                self.log_unified(label, payload)
            elif kind == "stream_token":
                # Posted as ("stream_token", token_str, (idx, total, tps)):
                # the token travels in the label slot, stats in the payload
                token_str = label
                token_idx, total_tokens, tps = payload
                self._handle_stream_token(token_str, token_idx, total_tokens, tps)
            else:
                self._handle_result(kind, label, payload)

    def _handle_stream_token(self, token_str, token_idx, total_tokens, tps):
        if self._stream_bubble is None:
            self._append_message("assistant", "")
            self._stream_bubble = getattr(self, "_last_bubble", None)
        if self._stream_bubble is not None:
            label = self._stream_bubble.text_label
            label.setText(label.text() + token_str)
            # Keep the stored message in sync so session switches keep the text
            session = self._sessions[self._current_session]
            if session["messages"]:
                session["messages"][-1]["text"] = label.text()
        self.status_label.setText(
            f"Generating... {token_idx}/{total_tokens} tokens ({tps:.1f} tok/s)"
        )
        self._scroll_chat_to_bottom()

    def set_busy(self, busy: bool, message: str = ""):
        self.busy = busy
        state = not busy
        for w in (
            self.train_btn, self.train_router_btn, self.send_btn, self.preview_btn,
            self.create_btn, self.delete_btn, self.new_chat_btn, self.refresh_models_btn,
            self.refresh_catalog_btn, self.model_combo,
        ):
            w.setEnabled(state)
        self.prompt_input.setEnabled(state)
        self.status_label.setText(message or ("Working..." if busy else "Ready"))

    # ---- result handling ----------------------------------------------------
