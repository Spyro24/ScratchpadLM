"""PySide6 UI mixin. Part of the `uiqt/` package split from `ui_qt.py`."""
import json
import os
import re
import time
from pathlib import Path

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor, QFont, QSyntaxHighlighter, QTextCharFormat, QTextCursor
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


class MarkerHighlighter(QSyntaxHighlighter):
    """Paints structural markers (`[EOL]`/`[EOS]`/`[EOT]`) yellow."""

    def __init__(self, document):
        super().__init__(document)
        self._format = QTextCharFormat()
        self._format.setForeground(QColor("#F7DC6F"))
        self._pattern = re.compile(r"\[(EOL|EOS|EOT)\]")

    def highlightBlock(self, text: str) -> None:
        for m in self._pattern.finditer(text):
            self.setFormat(m.start(), m.end() - m.start(), self._format)


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
        self._marker_highlighter = MarkerHighlighter(self.train_text.document())
        dg.addWidget(self.train_text)
        v.addWidget(data_group, 2)

        controls = QHBoxLayout()
        controls.addWidget(QLabel("Steps:"))
        self.steps_spin = QSpinBox()
        self.steps_spin.setRange(1, 1_000_000)
        self.steps_spin.setValue(self.cfg["training"]["steps"])
        controls.addWidget(self.steps_spin)
        controls.addWidget(QLabel("Batch size:"))
        self.batch_size_spin = QSpinBox()
        self.batch_size_spin.setRange(1, 512)
        self.batch_size_spin.setValue(int(self.cfg["training"]["batch_size"]))
        self.batch_size_spin.setToolTip("Windows per batch step (bigger = more GPU work)")
        controls.addWidget(self.batch_size_spin)
        controls.addWidget(QLabel("Batch steps:"))
        self.batch_steps_spin = QSpinBox()
        self.batch_steps_spin.setRange(1, 64)
        self.batch_steps_spin.setValue(int(self.cfg["training"].get("batch_steps", 1)))
        self.batch_steps_spin.setToolTip("Batches computed per weight update (gradient accumulation)")
        controls.addWidget(self.batch_steps_spin)
        self.use_gpu_cb = QCheckBox("GPU")
        self.use_gpu_cb.setChecked(bool(self.cfg["training"].get("use_gpu", True)))
        self.use_gpu_cb.setToolTip("Route worker gradients through CUDA when available (auto-fallback)")
        controls.addWidget(self.use_gpu_cb)
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
        self.auto_adjust_cb = QCheckBox("Auto-adjust")
        self.auto_adjust_cb.setChecked(bool(self.cfg["training"]["auto_adjust"]))
        controls.addWidget(self.auto_adjust_cb)
        controls.addStretch(1)
        v.addLayout(controls)

        # Auto-adjust settings panel (shows current values that auto-adjust can modify)
        self.auto_adjust_group = QGroupBox("Auto-Adjust Settings (Live)")
        aag = QGridLayout(self.auto_adjust_group)
        aag.setColumnStretch(1, 1)
        
        # LR multiplier
        aag.addWidget(QLabel("LR Multiplier:"), 0, 0)
        self.aa_lr_mult_label = QLabel("1.0000")
        self.aa_lr_mult_label.setStyleSheet(f"color: {p['accent']}; font-family: monospace;")
        aag.addWidget(self.aa_lr_mult_label, 0, 1)
        
        # Current momentum
        aag.addWidget(QLabel("Momentum:"), 0, 2)
        self.aa_momentum_label = QLabel("0.9000")
        self.aa_momentum_label.setStyleSheet(f"color: {p['accent']}; font-family: monospace;")
        aag.addWidget(self.aa_momentum_label, 0, 3)
        
        # Grad clip threshold
        aag.addWidget(QLabel("Grad Clip:"), 1, 0)
        self.aa_grad_clip_label = QLabel("1.0000")
        self.aa_grad_clip_label.setStyleSheet(f"color: {p['accent']}; font-family: monospace;")
        aag.addWidget(self.aa_grad_clip_label, 1, 1)
        
        # Reduce count
        aag.addWidget(QLabel("LR Reductions:"), 1, 2)
        self.aa_reduce_label = QLabel("0")
        self.aa_reduce_label.setStyleSheet(f"color: {p['accent']}; font-family: monospace;")
        aag.addWidget(self.aa_reduce_label, 1, 3)
        
        # Bad checks / plateau counter
        aag.addWidget(QLabel("Plateau Steps:"), 2, 0)
        self.aa_plateau_label = QLabel("0")
        self.aa_plateau_label.setStyleSheet(f"color: {p['muted']}; font-family: monospace;")
        aag.addWidget(self.aa_plateau_label, 2, 1)
        
        # Best loss
        aag.addWidget(QLabel("Best Loss:"), 2, 2)
        self.aa_best_loss_label = QLabel("—")
        self.aa_best_loss_label.setStyleSheet(f"color: {p['accent']}; font-family: monospace;")
        aag.addWidget(self.aa_best_loss_label, 2, 3)
        
        # Auto-adjust config (read-only display)
        aag.addWidget(QLabel("Auto Factor:"), 3, 0)
        self.aa_factor_label = QLabel(str(self.cfg["training"].get("auto_factor", 0.5)))
        aag.addWidget(self.aa_factor_label, 3, 1)
        
        aag.addWidget(QLabel("Auto Patience:"), 3, 2)
        self.aa_patience_label = QLabel(str(self.cfg["training"].get("auto_patience", 10)))
        aag.addWidget(self.aa_patience_label, 3, 3)
        
        aag.addWidget(QLabel("Cosine Restarts:"), 4, 0)
        self.aa_restart_label = QLabel("On" if self.cfg["training"].get("use_cosine_restarts", False) else "Off")
        aag.addWidget(self.aa_restart_label, 4, 1)
        
        aag.addWidget(QLabel("Restart Period:"), 4, 2)
        self.aa_restart_period_label = QLabel(str(self.cfg["training"].get("restart_period", 0)))
        aag.addWidget(self.aa_restart_period_label, 4, 3)
        
        v.addWidget(self.auto_adjust_group)

        prow = QHBoxLayout()
        self.train_progress = QProgressBar()
        self.train_progress.setValue(0)
        prow.addWidget(self.train_progress, 1)
        self.progress_label = QLabel("")
        self.progress_label.setMinimumWidth(220)
        prow.addWidget(self.progress_label)
        v.addLayout(prow)

        self.rollback_label = QLabel("Last rollback gap: —")
        v.addWidget(self.rollback_label)

        monitor = QGroupBox("Training Monitor")
        mg = QVBoxLayout(monitor)
        self.train_log = QPlainTextEdit()
        self.train_log.setReadOnly(True)
        self.train_log.setFont(MONO)
        mg.addWidget(self.train_log)
        v.addWidget(monitor, 2)
        return root

    # -- Model Info tab --
