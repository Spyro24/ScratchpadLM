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


class AdminTabMixin:
    def _build_admin_tab(self) -> QWidget:
        root = QWidget()
        v = QVBoxLayout(root)

        catalog_group = QGroupBox("Model Catalog")
        cg = QVBoxLayout(catalog_group)
        self.admin_list = QListWidget()
        self.admin_list.setFont(MONO)
        cg.addWidget(self.admin_list)
        v.addWidget(catalog_group, 1)
        self.refresh_catalog_btn = QPushButton("Refresh Catalog")
        self.refresh_catalog_btn.clicked.connect(self.refresh_catalog_list)
        v.addWidget(self.refresh_catalog_btn, alignment=Qt.AlignmentFlag.AlignHCenter)

        create_group = QGroupBox("Create Model")
        cr = QGridLayout(create_group)
        cr.addWidget(QLabel("Name:"), 0, 0)
        self.create_name_edit = QLineEdit()
        self.create_name_edit.setPlaceholderText("tiny-code")
        cr.addWidget(self.create_name_edit, 0, 1)
        cr.addWidget(QLabel("Domains:"), 0, 2)
        self.create_domains_edit = QLineEdit()
        self.create_domains_edit.setPlaceholderText("code, python")
        cr.addWidget(self.create_domains_edit, 0, 3)
        self.create_btn = QPushButton("Create")
        self.create_btn.clicked.connect(self.on_create_model)
        cr.addWidget(self.create_btn, 0, 4)
        v.addWidget(create_group)

        delete_group = QGroupBox("Delete Model")
        dr = QHBoxLayout(delete_group)
        self.delete_btn = QPushButton("Delete Selected")
        self.delete_btn.clicked.connect(self.on_delete_model)
        dr.addWidget(self.delete_btn)
        dr.addStretch(1)
        v.addWidget(delete_group)
        return root

    # -- Settings tab --
