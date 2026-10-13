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


class SettingsTabMixin:
    def _build_settings_tab(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("background: transparent;")
        body = QWidget()
        body.setStyleSheet("background: transparent;")
        grid = QGridLayout(body)
        grid.setColumnStretch(0, 0)
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(2, 0)
        grid.setColumnStretch(3, 1)

        settings = [
            ("generation", "max_tokens", "Max tokens", int),
            ("generation", "min_tokens", "Min tokens", int),
            ("generation", "temperature", "Temperature", float),
            ("generation", "top_k", "Top-k", int),
            ("generation", "top_p", "Top-p", float),
            ("generation", "repetition_penalty", "Repetition penalty", float),
            ("generation", "stop_token_ids", "Stop token IDs (comma-separated)", str),
            ("generation", "stop_strings", "Stop strings (comma-separated)", str),
            ("training", "steps", "Training steps", int),
            ("training", "lr", "Learning rate", float),
            ("training", "batch_size", "Batch size", int),
            ("training", "n_negatives", "Negatives", int),
            ("training", "workers", "Workers", int),
            ("training", "lr_scheduler", "LR scheduler", str),
            ("training", "warmup_steps", "Warmup steps", int),
            ("training", "min_lr", "Min learning rate", float),
            ("training", "grad_clip", "Grad clip norm", float),
            ("training", "val_split", "Validation split", float),
            ("training", "eval_every", "Eval every N steps", int),
            ("training", "early_stopping", "Early stopping", parse_bool),
            ("training", "patience", "Patience", int),
            ("training", "min_delta", "Min delta", float),
            ("training", "checkpoint_every", "Checkpoint every N", int),
            ("training", "checkpoint_dir", "Checkpoint dir", str),
            ("training", "auto_adjust", "Auto LR adjust", parse_bool),
            ("training", "auto_factor", "Auto LR factor", float),
            ("training", "auto_patience", "Auto LR patience", int),
            ("training", "auto_clip_relax", "Auto clip relax", parse_bool),
            ("training", "momentum", "Momentum", float),
            ("tokenizer", "vocab_size", "Tokenizer vocab size", int),
            ("tokenizer", "min_frequency", "Min frequency", int),
            ("model", "vocab_size", "Model vocab size", int),
            ("model", "ctx", "Context length", int),
            ("model", "emb", "Embedding dim", int),
            ("model", "hidden", "Hidden dim", int),
            ("ui", "theme", "UI Theme", str),
        ]

        # Two-column layout of labeled entries, section headers in bold
        col = 0
        row = 0
        current_section = None
        half = len(settings) // 2 + 1
        for idx, (section, key, label, cast) in enumerate(settings):
            if idx == half:
                col = 2
                row = 0
            if section != current_section:
                current_section = section
                header = QLabel(section.upper())
                header.setStyleSheet("font-weight: bold;")
                grid.addWidget(header, row, col, 1, 2)
                row += 1
            
            # Special handling for theme setting - use QComboBox
            if section == "ui" and key == "theme":
                grid.addWidget(QLabel(label + ":"), row, col)
                self.theme_combo = QComboBox()
                self.theme_combo.addItems(["light", "dark", "system"])
                self.theme_combo.setCurrentText(str(self.cfg[section][key]))
                self.theme_combo.currentTextChanged.connect(self._on_theme_change)
                grid.addWidget(self.theme_combo, row, col + 1)
                self._settings_entries[(section, key)] = (self.theme_combo, cast)
            else:
                grid.addWidget(QLabel(label + ":"), row, col)
                edit = QLineEdit(str(self.cfg[section][key]))
                grid.addWidget(edit, row, col + 1)
                self._settings_entries[(section, key)] = (edit, cast)
            row += 1

        # ---- Config Editor Section ----
        row = max(row, half + 5) + 1
        config_header = QLabel("CONFIG EDITOR")
        config_header.setStyleSheet("font-weight: bold;")
        grid.addWidget(config_header, row, 0, 1, 4)
        row += 1

        self.config_editor = QPlainTextEdit()
        self.config_editor.setFont(MONO)
        self.config_editor.setMinimumHeight(200)
        self.config_editor.setPlaceholderText("YAML config editor with validation...")
        grid.addWidget(self.config_editor, row, 0, 1, 4)
        row += 1

        config_btn_row = QHBoxLayout()
        config_load_btn = QPushButton("Load Current Config")
        config_load_btn.clicked.connect(self.on_load_config_editor)
        config_validate_btn = QPushButton("Validate Config")
        config_validate_btn.clicked.connect(self.on_validate_config_editor)
        config_save_btn = QPushButton("Save Config from Editor")
        config_save_btn.clicked.connect(self.on_save_config_editor)
        config_btn_row.addWidget(config_load_btn)
        config_btn_row.addWidget(config_validate_btn)
        config_btn_row.addWidget(config_save_btn)
        config_btn_row.addStretch(1)
        config_btn_wrap = QWidget()
        config_btn_wrap.setLayout(config_btn_row)
        grid.addWidget(config_btn_wrap, row, 0, 1, 4)
        row += 1

        # Preset management
        preset_row = QHBoxLayout()
        self.preset_combo = QComboBox()
        self.preset_combo.addItems(["Default", "Fast Training", "High Quality", "Low Memory", "Custom..."])
        self.preset_combo.currentTextChanged.connect(self.on_preset_changed)
        preset_row.addWidget(QLabel("Preset:"))
        preset_row.addWidget(self.preset_combo)
        preset_save_btn = QPushButton("Save as Preset")
        preset_save_btn.clicked.connect(self.on_save_preset)
        preset_row.addWidget(preset_save_btn)
        preset_row.addStretch(1)
        preset_wrap = QWidget()
        preset_wrap.setLayout(preset_row)
        grid.addWidget(preset_wrap, row, 0, 1, 4)
        row += 1

        # Tokenizer manager section (spans both columns)
        row += 1
        tok_header = QLabel("TOKENIZER MANAGER")
        tok_header.setStyleSheet("font-weight: bold;")
        grid.addWidget(tok_header, row, 0, 1, 4)
        row += 1

        grid.addWidget(QLabel("Current Vocab Size:"), row, 0)
        self.tok_vocab_label = QLabel(str(len(self.tok.vocab)))
        self.tok_vocab_label.setFont(MONO)
        grid.addWidget(self.tok_vocab_label, row, 1)
        row += 1

        grid.addWidget(QLabel("Learned Merges:"), row, 0)
        self.tok_merges_label = QLabel(str(len(self.tok.merges)))
        self.tok_merges_label.setFont(MONO)
        grid.addWidget(self.tok_merges_label, row, 1)
        row += 1

        grid.addWidget(QLabel("Extend with text:"), row, 0, Qt.AlignmentFlag.AlignTop)
        self.tok_extend_text = QPlainTextEdit()
        self.tok_extend_text.setPlaceholderText("Text to extend the vocabulary with…")
        self.tok_extend_text.setMaximumHeight(100)
        grid.addWidget(self.tok_extend_text, row, 1, 1, 3)
        row += 1

        extend_row = QHBoxLayout()
        extend_row.addWidget(QLabel("Target vocab size:"))
        self.tok_target_vocab_edit = QLineEdit(str(self.cfg["tokenizer"]["vocab_size"]))
        self.tok_target_vocab_edit.setMaximumWidth(90)
        extend_row.addWidget(self.tok_target_vocab_edit)
        extend_row.addWidget(QLabel("Min freq:"))
        self.tok_min_freq_edit = QLineEdit(str(self.cfg["tokenizer"]["min_frequency"]))
        self.tok_min_freq_edit.setMaximumWidth(70)
        extend_row.addWidget(self.tok_min_freq_edit)
        extend_btn = QPushButton("Extend Vocab")
        extend_btn.clicked.connect(self.on_extend_vocab)
        extend_row.addWidget(extend_btn)
        extend_row.addStretch(1)
        wrap = QWidget()
        wrap.setLayout(extend_row)
        grid.addWidget(wrap, row, 1, 1, 3)
        row += 1

        grid.addWidget(QLabel("Learned Patterns (sample):"), row, 0, Qt.AlignmentFlag.AlignTop)
        self.tok_patterns_text = QPlainTextEdit()
        self.tok_patterns_text.setReadOnly(True)
        self.tok_patterns_text.setFont(MONO)
        grid.addWidget(self.tok_patterns_text, row, 1, 1, 3)
        row += 1

        patterns_btn = QPushButton("Refresh Patterns")
        patterns_btn.clicked.connect(self.refresh_tokenizer_patterns)
        grid.addWidget(patterns_btn, row, 1, alignment=Qt.AlignmentFlag.AlignLeft)
        row += 1

        save_btn = QPushButton("Save Config")
        save_btn.clicked.connect(self.on_save_config)
        grid.addWidget(save_btn, row, 0, 1, 4, Qt.AlignmentFlag.AlignHCenter)

        scroll.setWidget(body)
        self.refresh_tokenizer_patterns()
        return scroll

    # -- Logs tab --
