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


class TrainingTabMixin:
    def _build_training_tab(self) -> QWidget:
        root = QWidget()
        v = QVBoxLayout(root)
        p = PALETTE

        target_row = QHBoxLayout()
        t = QLabel("Training target:")
        t.setStyleSheet("font-weight: bold;")
        target_row.addWidget(t)
        self.target_label = QLabel("TinyLM")
        self.target_label.setStyleSheet(f"color: {p['accent']};")
        target_row.addWidget(self.target_label)
        target_row.addStretch(1)
        v.addLayout(target_row)

        data_group = QGroupBox("Training Data")
        dg = QVBoxLayout(data_group)
        self.train_text = QPlainTextEdit()
        self.train_text.setPlaceholderText("Enter training text here…")
        self.train_text.setMinimumHeight(120)
        dg.addWidget(self.train_text)
        v.addWidget(data_group, 2)

        controls = QHBoxLayout()
        controls.addWidget(QLabel("Steps:"))
        self.steps_spin = QSpinBox()
        self.steps_spin.setRange(1, 1_000_000)
        self.steps_spin.setValue(self.cfg["training"]["steps"])
        controls.addWidget(self.steps_spin)
        self.train_btn = QPushButton("Train on Data")
        self.train_btn.clicked.connect(self.on_train)
        controls.addWidget(self.train_btn)
        self.train_router_btn = QPushButton("Train Router")
        self.train_router_btn.clicked.connect(self.on_train_router)
        controls.addWidget(self.train_router_btn)
        self.cancel_train_btn = QPushButton("Cancel Training")
        self.cancel_train_btn.setEnabled(False)
        self.cancel_train_btn.clicked.connect(self.on_cancel_training)
        controls.addWidget(self.cancel_train_btn)
        controls.addStretch(1)
        v.addLayout(controls)

        prow = QHBoxLayout()
        self.train_progress = QProgressBar()
        self.train_progress.setValue(0)
        prow.addWidget(self.train_progress, 1)
        self.progress_label = QLabel("")
        self.progress_label.setMinimumWidth(220)
        prow.addWidget(self.progress_label)
        v.addLayout(prow)

        monitor = QGroupBox("Training Monitor")
        mg = QVBoxLayout(monitor)
        self.train_log = QPlainTextEdit()
        self.train_log.setReadOnly(True)
        self.train_log.setFont(MONO)
        mg.addWidget(self.train_log)
        v.addWidget(monitor, 2)
        return root

    # -- Model Info tab --
