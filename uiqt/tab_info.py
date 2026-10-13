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


class InfoTabMixin:
    def _build_model_info_tab(self) -> QWidget:
        root = QWidget()
        v = QVBoxLayout(root)

        group = QGroupBox("Model Inspector")
        form = QFormLayout(group)
        self.inspector_model_name = QLabel("None")
        self.inspector_arch = QLabel("Unknown")
        self.inspector_params = QLabel("0")
        self.inspector_memory = QLabel("0 KB")
        self.inspector_vocab = QLabel("0")
        self.inspector_ctx = QLabel("0")
        self.inspector_emb = QLabel("0")
        self.inspector_hidden = QLabel("0")
        for widget in (
            self.inspector_model_name, self.inspector_arch, self.inspector_params,
            self.inspector_memory, self.inspector_vocab, self.inspector_ctx,
            self.inspector_emb, self.inspector_hidden,
        ):
            widget.setFont(MONO)
        form.addRow("Model:", self.inspector_model_name)
        form.addRow("Architecture:", self.inspector_arch)
        form.addRow("Parameters:", self.inspector_params)
        form.addRow("Memory (4-bit):", self.inspector_memory)
        form.addRow("Vocab Size:", self.inspector_vocab)
        form.addRow("Context:", self.inspector_ctx)
        form.addRow("Embedding:", self.inspector_emb)
        form.addRow("Hidden:", self.inspector_hidden)
        v.addWidget(group)

        btns = QHBoxLayout()
        self.tensor_detail_btn = QPushButton("Show Tensor Details")
        self.tensor_detail_btn.setEnabled(False)
        self.tensor_detail_btn.clicked.connect(self.on_show_tensor_details)
        btns.addWidget(self.tensor_detail_btn)
        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self.update_model_inspector)
        btns.addWidget(refresh_btn)
        btns.addStretch(1)
        v.addLayout(btns)
        v.addStretch(1)
        return root

    # -- Admin tab --
