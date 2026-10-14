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
    QToolTip,
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

        # Define settings with metadata: (section, key, label, widget_type, options)
        # widget_type: "spin_int", "spin_float", "combo", "check", "edit", "dir"
        settings = [
            # Generation settings
            ("generation", "max_tokens", "Max tokens (0=∞)", "spin_int", {"min": 0, "max": 100000, "step": 1, "tooltip": "Maximum tokens to generate (0 = unlimited, structural markers end generation)"}),
            ("generation", "min_tokens", "Min tokens", "spin_int", {"min": 0, "max": 100000, "step": 1, "tooltip": "Minimum tokens to generate before allowing stop"}),
            ("generation", "temperature", "Temperature", "spin_float", {"min": 0.01, "max": 5.0, "step": 0.05, "tooltip": "Sampling temperature (higher = more random)"}),
            ("generation", "top_k", "Top-k", "spin_int", {"min": 0, "max": 10000, "step": 1, "tooltip": "Top-k sampling (0 = disabled)"}),
            ("generation", "top_p", "Top-p (nucleus)", "spin_float", {"min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "Nucleus sampling threshold (0 = disabled)"}),
            ("generation", "repetition_penalty", "Repetition penalty", "spin_float", {"min": 1.0, "max": 2.0, "step": 0.05, "tooltip": "Penalty for repeating tokens (1.0 = off)"}),
            ("generation", "stop_token_ids", "Stop token IDs (comma-separated)", "edit", {"tooltip": "Comma-separated token IDs that stop generation"}),
            ("generation", "stop_strings", "Stop strings (comma-separated)", "edit", {"tooltip": "Comma-separated strings that stop generation"}),
            
            # Training settings
            ("training", "steps", "Training steps", "spin_int", {"min": 1, "max": 1000000, "step": 1, "tooltip": "Number of training steps"}),
            ("training", "lr", "Learning rate", "spin_float", {"min": 1e-6, "max": 1.0, "step": 0.001, "tooltip": "Base learning rate"}),
            ("training", "batch_size", "Batch size", "spin_int", {"min": 1, "max": 512, "step": 1, "tooltip": "Mini-batch size"}),
            ("training", "n_negatives", "Negative samples", "spin_int", {"min": 0, "max": 2048, "step": 1, "tooltip": "Number of negative samples for sampled softmax"}),
            ("training", "workers", "Workers (parallel)", "spin_int", {"min": 1, "max": 64, "step": 1, "tooltip": "Parallel workers (None = auto)"}),
            ("training", "lr_scheduler", "LR scheduler", "combo", {"items": ["cosine_warmup", "linear", "constant"], "tooltip": "Learning rate schedule"}),
            ("training", "warmup_steps", "Warmup steps", "spin_int", {"min": 0, "max": 10000, "step": 1, "tooltip": "Linear warmup steps"}),
            ("training", "min_lr", "Min learning rate", "spin_float", {"min": 0.0, "max": 0.1, "step": 0.0001, "tooltip": "Minimum learning rate floor"}),
            ("training", "grad_clip", "Grad clip norm", "spin_float", {"min": 0.0, "max": 10.0, "step": 0.1, "tooltip": "Gradient clipping by global norm (0 = off)"}),
            ("training", "val_split", "Validation split", "spin_float", {"min": 0.0, "max": 0.5, "step": 0.05, "tooltip": "Fraction of data for validation"}),
            ("training", "eval_every", "Eval every N steps", "spin_int", {"min": 0, "max": 1000, "step": 1, "tooltip": "Validation frequency (0 = off)"}),
            ("training", "early_stopping", "Early stopping", "check", {"tooltip": "Stop training when validation plateaus"}),
            ("training", "patience", "Patience", "spin_int", {"min": 1, "max": 100, "step": 1, "tooltip": "Early stopping patience"}),
            ("training", "min_delta", "Min delta", "spin_float", {"min": 1e-6, "max": 0.1, "step": 1e-4, "tooltip": "Minimum improvement threshold"}),
            ("training", "checkpoint_every", "Checkpoint every N", "spin_int", {"min": 0, "max": 10000, "step": 1, "tooltip": "Save checkpoint frequency (0 = off)"}),
            ("training", "checkpoint_dir", "Checkpoint dir", "dir", {"tooltip": "Directory for checkpoints"}),
            ("training", "auto_adjust", "Auto LR adjust", "check", {"tooltip": "Automatically reduce LR on plateaus"}),
            ("training", "auto_factor", "Auto LR factor", "spin_float", {"min": 0.1, "max": 0.9, "step": 0.05, "tooltip": "LR reduction factor on plateau"}),
            ("training", "auto_patience", "Auto LR patience", "spin_int", {"min": 1, "max": 100, "step": 1, "tooltip": "Plateau patience for auto-adjust"}),
            ("training", "auto_clip_relax", "Auto clip relax", "check", {"tooltip": "Relax grad clip when chronically clipping"}),
            ("training", "momentum", "Momentum", "spin_float", {"min": 0.0, "max": 0.999, "step": 0.01, "tooltip": "SGD momentum (0 = plain SGD)"}),
            ("training", "rollback_window", "Rollback window", "spin_int", {"min": 0, "max": 100, "step": 1, "tooltip": "Loss spike rollback window (0 = off)"}),
            ("training", "optimizer", "Optimizer", "combo", {"items": ["sgd", "apollo", "apollo_mini", "adam", "lion", "muon"], "tooltip": "Optimizer: adam/muon reduce loss fastest, lion is memory-light, apollo balances both"}),
            ("training", "batch_steps", "Batch steps", "spin_int", {"min": 1, "max": 64, "step": 1, "tooltip": "Batches computed per weight update (more GPU work per step)"}),
            ("training", "use_gpu", "Use GPU", "check", {"tooltip": "Route worker gradients through CUDA when available (auto-fallback)"}),
            
            # New training features
            ("training", "use_frequency_neg_sampling", "Freq. neg sampling", "check", {"tooltip": "Use unigram^0.75 weighted negative sampling"}),
            ("training", "n_negatives_start", "Neg. start (adaptive)", "spin_int", {"min": 0, "max": 2048, "step": 1, "tooltip": "Initial n_negatives for adaptive decay (None = auto)"}),
            ("training", "n_negatives_end", "Neg. end (adaptive)", "spin_int", {"min": 0, "max": 2048, "step": 1, "tooltip": "Final n_negatives for adaptive decay"}),
            ("training", "n_negatives_decay", "Neg. decay schedule", "combo", {"items": ["exponential", "linear", "cosine"], "tooltip": "Adaptive n_negatives decay schedule"}),
            ("training", "use_cosine_restarts", "Cosine restarts (SGDR)", "check", {"tooltip": "Cosine annealing with warm restarts"}),
            ("training", "restart_period", "Restart period", "spin_int", {"min": 0, "max": 10000, "step": 1, "tooltip": "Cosine restart period (0 = auto)"}),
            ("training", "cyclic_lr", "Cyclic LR", "check", {"tooltip": "Enable cyclic learning rates"}),
            ("training", "cycle_length", "Cycle length", "spin_int", {"min": 0, "max": 10000, "step": 1, "tooltip": "Cyclic LR period (0 = auto)"}),
            
            # Tokenizer settings
            ("tokenizer", "vocab_size", "Tokenizer vocab size", "spin_int", {"min": 256, "max": 100000, "step": 1, "tooltip": "Target vocabulary size"}),
            ("tokenizer", "min_frequency", "Min frequency", "spin_int", {"min": 1, "max": 100, "step": 1, "tooltip": "Minimum token frequency for merges"}),
            
            # Model settings
            ("model", "vocab_size", "Model vocab size", "spin_int", {"min": 256, "max": 100000, "step": 1, "tooltip": "Model vocabulary size"}),
            ("model", "ctx", "Context length", "spin_int", {"min": 1, "max": 1024, "step": 1, "tooltip": "Context window size"}),
            ("model", "emb", "Embedding dim", "spin_int", {"min": 1, "max": 1024, "step": 1, "tooltip": "Token embedding dimension"}),
            ("model", "hidden", "Hidden dim", "spin_int", {"min": 1, "max": 2048, "step": 1, "tooltip": "Hidden layer dimension"}),
            
            # UI settings
            ("ui", "theme", "UI Theme", "combo", {"items": ["light", "dark", "system"], "tooltip": "Application theme"}),
        ]

        # Two-column layout of labeled entries, section headers in bold
        col = 0
        row = 0
        current_section = None
        half = len(settings) // 2 + 1
        for idx, (section, key, label, widget_type, opts) in enumerate(settings):
            if idx == half:
                col = 2
                row = 0
            if section != current_section:
                current_section = section
                header = QLabel(section.upper())
                header.setStyleSheet("font-weight: bold; color: #2C3E50;")
                grid.addWidget(header, row, col, 1, 2)
                row += 1
            
            label_widget = QLabel(label + ":")
            label_widget.setToolTip(opts.get("tooltip", ""))
            grid.addWidget(label_widget, row, col)
            
            # Create appropriate widget based on type
            if widget_type == "spin_int":
                widget = QSpinBox()
                widget.setRange(opts.get("min", 0), opts.get("max", 1000000))
                widget.setSingleStep(opts.get("step", 1))
                val = self.cfg[section][key]
                if val is None:
                    # Use 0 to represent "auto" for optional fields
                    widget.setValue(0)
                    widget.setSpecialValueText("Auto")
                else:
                    widget.setValue(int(val))
                widget.setToolTip(opts.get("tooltip", ""))
            elif widget_type == "spin_float":
                widget = QDoubleSpinBox()
                widget.setRange(opts.get("min", 0.0), opts.get("max", 1.0))
                widget.setSingleStep(opts.get("step", 0.01))
                widget.setDecimals(6)
                val = self.cfg[section][key]
                widget.setValue(float(val) if val is not None else 0.0)
                widget.setToolTip(opts.get("tooltip", ""))
            elif widget_type == "combo":
                widget = QComboBox()
                widget.addItems(opts.get("items", []))
                current_val = str(self.cfg[section][key])
                idx_combo = widget.findText(current_val)
                if idx_combo >= 0:
                    widget.setCurrentIndex(idx_combo)
                widget.setToolTip(opts.get("tooltip", ""))
            elif widget_type == "check":
                widget = QCheckBox()
                widget.setChecked(bool(self.cfg[section][key]))
                widget.setToolTip(opts.get("tooltip", ""))
            elif widget_type == "dir":
                widget = QLineEdit(str(self.cfg[section][key]))
                widget.setToolTip(opts.get("tooltip", ""))
                # Add browse button
                browse_btn = QPushButton("…")
                browse_btn.setMaximumWidth(30)
                browse_btn.setToolTip("Browse for directory")
                def make_browse(w=widget):
                    return lambda: self._browse_dir(w)
                browse_btn.clicked.connect(make_browse())
                hbox = QHBoxLayout()
                hbox.setContentsMargins(0, 0, 0, 0)
                hbox.addWidget(widget)
                hbox.addWidget(browse_btn)
                widget_wrapper = QWidget()
                widget_wrapper.setLayout(hbox)
                widget = widget_wrapper
            else:  # edit
                widget = QLineEdit(str(self.cfg[section][key]))
                widget.setToolTip(opts.get("tooltip", ""))
            
            grid.addWidget(widget, row, col + 1)
            self._settings_entries[(section, key)] = (widget, None)  # Cast handled in on_save_config
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
