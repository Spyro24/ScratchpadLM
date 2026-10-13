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


class LogsTabMixin:
    def _build_logs_tab(self) -> QWidget:
        root = QWidget()
        v = QVBoxLayout(root)
        p = PALETTE

        filters = QGroupBox("Filters")
        f = QHBoxLayout(filters)
        self.log_filter_boxes = {}
        log_types = [
            ("training", "Training", p["accent"]),
            ("generation", "Generation", p["success"]),
            ("system", "System", p["muted"]),
            ("error", "Error", p["error"]),
        ]
        for key, label, color in log_types:
            cb = QCheckBox(label)
            cb.setChecked(True)
            cb.setStyleSheet(f"color: {color};")
            cb.stateChanged.connect(self._apply_log_filters)
            self.log_filter_boxes[key] = cb
            f.addWidget(cb)
        f.addSpacing(16)
        f.addWidget(QLabel("Search:"))
        self.log_search_edit = QLineEdit()
        self.log_search_edit.setPlaceholderText("filter text…")
        self.log_search_edit.setMaximumWidth(220)
        self.log_search_edit.textChanged.connect(self._apply_log_filters)
        f.addWidget(self.log_search_edit)
        self.log_autoscroll_cb = QCheckBox("Auto-scroll")
        self.log_autoscroll_cb.setChecked(True)
        f.addWidget(self.log_autoscroll_cb)
        f.addStretch(1)
        clear_btn = QPushButton("Clear Logs")
        clear_btn.clicked.connect(self._clear_unified_logs)
        f.addWidget(clear_btn)
        export_btn = QPushButton("Export Logs")
        export_btn.clicked.connect(self._export_logs)
        f.addWidget(export_btn)
        v.addWidget(filters)

        log_group = QGroupBox("Unified Logs")
        lg = QVBoxLayout(log_group)
        self.unified_log = QTextEdit()
        self.unified_log.setReadOnly(True)
        self.unified_log.setFont(MONO)
        lg.addWidget(self.unified_log)
        v.addWidget(log_group, 1)
        return root

    # ---- chat sessions & bubbles -------------------------------------------
