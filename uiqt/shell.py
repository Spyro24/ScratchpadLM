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
from tinylm.training import _cuda_available, _accel_available
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


class ShellMixin:
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        outer = QVBoxLayout(central)
        outer.setContentsMargins(8, 8, 8, 4)

        self.tabs = QTabWidget()
        outer.addWidget(self.tabs)
        self.tabs.addTab(self._build_chat_tab(), "Chat")
        self.tabs.addTab(self._build_training_tab(), "Training")
        self.tabs.addTab(self._build_tasks_tab(), "Tasks")
        self.tabs.addTab(self._build_model_info_tab(), "Model Info")
        self.tabs.addTab(self._build_admin_tab(), "Admin")
        self.tabs.addTab(self._build_settings_tab(), "Settings")
        self.tabs.addTab(self._build_logs_tab(), "Logs")

        self.status_label = QLabel("Ready")
        sb = QStatusBar()
        sb.addWidget(self.status_label, 1)
        
        # Determine which kernel will be used at startup
        use_c = self.cfg["training"].get("use_c", None)
        advanced_features = (
            False or  # val_ids
            0.0 > 0 or  # val_split
            0 > 0 or  # eval_every
            False or  # early_stopping
            0 > 0 or  # checkpoint_every
            False or  # cancel_callback
            False  # optimizer in ("apollo", "apollo_mini")
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
        
        # Kernel indicator (permanent widget on the right)
        self.kernel_indicator = QLabel(f"Kernel: {kernel_name}")
        self.kernel_indicator.setStyleSheet("color: #888; padding-right: 8px;")
        sb.addPermanentWidget(self.kernel_indicator)
        
        self.setStatusBar(sb)

    # -- Chat tab (sidebar list | chat area | input + send) --
