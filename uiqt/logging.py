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

from ai import (
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


class LoggingMixin:
    def log_train(self, message: str, is_error: bool = False):
        self._log_train(message, is_error)

    def _log_train(self, message: str, is_error: bool = False):
        ts = time.strftime("%H:%M:%S")
        color = PALETTE["error"] if is_error else PALETTE["text"]
        self.train_log.appendHtml(
            f'<span style="color:{PALETTE["muted"]}">[{ts}]</span> '
            f'<span style="color:{color}">{self._escape(message)}</span>'
        )
        self.log_unified("error" if is_error else "training", message)

    def log_unified(self, log_type: str, message: str):
        ts = time.strftime("%H:%M:%S")
        self._all_log_entries.append((ts, log_type, message))
        if len(self._all_log_entries) > 10000:
            self._all_log_entries = self._all_log_entries[-10000:]
        self._apply_log_filters()

    def _apply_log_filters(self, event=None):
        search = self.log_search_edit.text().strip().lower()
        show = {k for k, cb in self.log_filter_boxes.items() if cb.isChecked()}
        html = []
        colors = {
            "training": PALETTE["accent"],
            "generation": PALETTE["success"],
            "system": PALETTE["muted"],
            "error": PALETTE["error"],
        }
        for ts, log_type, message in self._all_log_entries:
            if log_type not in show:
                continue
            if search and search not in message.lower() and search not in ts.lower():
                continue
            color = colors.get(log_type, PALETTE["text"])
            kind = log_type.upper()
            html.append(
                f'<span style="color:#95A5A6">[{ts}]</span> '
                f'<b><span style="color:{color}">[{kind}]</span></b> '
                f'<span style="color:{color}">{self._escape(message)}</span>'
            )
        self.unified_log.setHtml("<br>".join(html))
        if self.log_autoscroll_cb.isChecked():
            self.unified_log.moveCursor(QTextCursor.MoveOperation.End)

    def _clear_unified_logs(self):
        self._all_log_entries.clear()
        self.unified_log.clear()

    def _export_logs(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Logs", "logs.txt", "Text files (*.txt);;All files (*)"
        )
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as f:
                for ts, log_type, message in self._all_log_entries:
                    f.write(f"[{ts}] [{log_type.upper()}] {message}\n")
            self.log_unified("system", f"Logs exported to {path}")
        except Exception as exc:
            self.log_unified("error", f"Failed to export logs: {exc}")

    @staticmethod

    def _escape(text: str) -> str:
        return (
            str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        )

    # ---- refresh helpers ------------------------------------------------------
