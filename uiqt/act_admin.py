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


class AdminActionsMixin:
    def on_create_model(self):
        if self.busy:
            return
        name = self.create_name_edit.text().strip()
        if not name:
            self.show_warning("No name", "Enter a model name (e.g. tiny-code).")
            return
        if MODEL_CATALOG.get(name) is not None:
            self.show_warning("Exists", f"Model '{name}' already exists in the catalog.")
            return
        domains_raw = self.create_domains_edit.text().strip()
        domains = [d.strip() for d in domains_raw.split(",") if d.strip()] or [name]
        base_config = {
            "ctx": self.cfg["model"]["ctx"],
            "emb": self.cfg["model"]["emb"],
            "hidden": self.cfg["model"]["hidden"],
        }
        self.log_unified("system", f"Creating model '{name}' (domains: {', '.join(domains)})...")
        self.worker.submit("Creating model", self._create_model_job, name, domains, base_config)
        self.set_busy(True, "Creating model...")

    def _create_model_job(self, name, domains, base_config):
        creator = ModelCreator(MODEL_CATALOG, "models", tokenizer_manager=TOKENIZER_MANAGER)
        return creator.create_model(name, domains, base_config=base_config)

    def on_delete_model(self):
        if self.busy:
            return
        row = self.admin_list.currentRow()
        if row < 0 or row >= len(self._catalog_names):
            self.show_warning("No selection", "Select a model in the catalog list first.")
            return
        name = self._catalog_names[row]
        if not self.ask_yes_no(
            "Confirm delete",
            f"Delete model '{name}' from the catalog?\n(The .tlm file is removed too.)",
        ):
            return
        self.worker.submit("Deleting model", self._delete_model_job, name)
        self.set_busy(True, "Deleting model...")

    def _delete_model_job(self, name):
        spec = MODEL_CATALOG.get(name)
        removed = MODEL_CATALOG.remove(name)
        if spec is not None and os.path.exists(spec.path):
            try:
                os.remove(spec.path)
            except OSError:
                pass
        return name, removed
