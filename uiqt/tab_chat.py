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


class ChatTabMixin:
    def _build_chat_tab(self) -> QWidget:
        root = QWidget()
        lay = QHBoxLayout(root)
        lay.setContentsMargins(4, 4, 4, 4)
        p = PALETTE

        splitter = QSplitter(Qt.Orientation.Horizontal)
        lay.addWidget(splitter)

        # Left: chats list sidebar
        side = QWidget()
        side_l = QVBoxLayout(side)
        side_l.setContentsMargins(0, 0, 6, 0)
        title = QLabel("Chats")
        title.setStyleSheet("font-weight: bold; font-size: 14px;")
        side_l.addWidget(title)
        self.chat_list = QListWidget()
        self.chat_list.currentRowChanged.connect(self._on_session_selected)
        side_l.addWidget(self.chat_list, 1)
        self.new_chat_btn = QPushButton("＋ New Chat")
        self.new_chat_btn.clicked.connect(self._new_session)
        side_l.addWidget(self.new_chat_btn)
        side.setMinimumWidth(180)
        side.setMaximumWidth(260)
        splitter.addWidget(side)

        # Center: model row, chat area, gen settings, input + send
        center = QWidget()
        c = QVBoxLayout(center)
        c.setContentsMargins(0, 0, 0, 0)

        mrow = QHBoxLayout()
        mrow.addWidget(QLabel("Model:"))
        self.model_combo = QComboBox()
        self.model_combo.setMinimumWidth(180)
        mrow.addWidget(self.model_combo)
        self.refresh_models_btn = QPushButton("Refresh Models")
        self.refresh_models_btn.clicked.connect(self.refresh_models)
        mrow.addWidget(self.refresh_models_btn)
        # Use selected model instead of router
        self.use_selected_model_cb = QCheckBox("Use selected model (bypass router)")
        self.use_selected_model_cb.setToolTip("When checked, use the manually selected model instead of letting the router choose")
        mrow.addWidget(self.use_selected_model_cb)
        self.vocab_label = QLabel(f"Vocab: {len(self.tok.vocab)}")
        self.vocab_label.setStyleSheet(f"color: {p['muted']};")
        mrow.addWidget(self.vocab_label)
        mrow.addStretch(1)
        c.addLayout(mrow)

        # Chat area: scrollable column of bubbles
        self.chat_scroll = QScrollArea()
        self.chat_scroll.setWidgetResizable(True)
        self.chat_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.chat_scroll.setStyleSheet("background: transparent;")
        self._chat_host = QWidget()
        self._chat_host.setStyleSheet("background: transparent;")
        self._chat_vbox = QVBoxLayout(self._chat_host)
        self._chat_vbox.setContentsMargins(6, 6, 6, 6)
        self._chat_vbox.addStretch(1)
        self.chat_scroll.setWidget(self._chat_host)
        c.addWidget(self.chat_scroll, 1)

        # Quick generation settings
        gen = QGroupBox("Generation Settings")
        g = QHBoxLayout(gen)
        self.max_tokens_spin = QSpinBox()
        self.max_tokens_spin.setRange(0, 10000)
        self.max_tokens_spin.setSpecialValueText("∞")
        self.min_tokens_spin = QSpinBox()
        self.min_tokens_spin.setRange(0, 10000)
        self.temperature_spin = QDoubleSpinBox()
        self.temperature_spin.setRange(0.05, 5.0)
        self.temperature_spin.setSingleStep(0.05)
        self.top_k_spin = QSpinBox()
        self.top_k_spin.setRange(0, 10000)
        self.top_p_spin = QDoubleSpinBox()
        self.top_p_spin.setRange(0.0, 1.0)
        self.top_p_spin.setSingleStep(0.05)
        self.rep_penalty_spin = QDoubleSpinBox()
        self.rep_penalty_spin.setRange(1.0, 2.0)
        self.rep_penalty_spin.setSingleStep(0.05)
        # Stop token IDs (comma-separated)
        self.stop_tokens_edit = QLineEdit()
        self.stop_tokens_edit.setPlaceholderText("Stop token IDs (comma-separated)")
        self.stop_tokens_edit.setMaximumWidth(150)
        # Stop strings (comma-separated)
        self.stop_strings_edit = QLineEdit()
        self.stop_strings_edit.setPlaceholderText("Stop strings (comma-separated)")
        self.stop_strings_edit.setMaximumWidth(150)
        for label, widget in (
            ("Max Tokens (0=∞)", self.max_tokens_spin),
            ("Min Tokens", self.min_tokens_spin),
            ("Temp", self.temperature_spin),
            ("Top-k", self.top_k_spin),
            ("Top-p", self.top_p_spin),
            ("Rep. penalty", self.rep_penalty_spin),
            ("Stop IDs", self.stop_tokens_edit),
            ("Stop Strings", self.stop_strings_edit),
        ):
            g.addWidget(QLabel(label))
            g.addWidget(widget)
        self._load_gen_settings()
        self.preview_btn = QPushButton("Generate with Preview")
        self.preview_btn.setToolTip("Show detailed timing (tokens/sec) after generation")
        self.preview_btn.clicked.connect(self.on_generate_preview)
        g.addWidget(self.preview_btn)
        g.addStretch(1)
        c.addWidget(gen)

        # Input row: text field + send button on the right
        irow = QHBoxLayout()
        self.prompt_input = QLineEdit()
        self.prompt_input.setPlaceholderText("Type a prompt and press Enter…")
        self.prompt_input.returnPressed.connect(self.on_send_prompt)
        self.send_btn = QPushButton("Send")
        self.send_btn.setObjectName("sendButton")
        self.send_btn.clicked.connect(self.on_send_prompt)
        irow.addWidget(self.prompt_input, 1)
        irow.addWidget(self.send_btn)
        c.addLayout(irow)

        splitter.addWidget(center)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([220, 960])

        # Connect model selection after the combo is populated
        self.model_combo.currentTextChanged.connect(self._on_model_change)
        return root

    def _load_gen_settings(self):
        gen = self.cfg["generation"]
        self.max_tokens_spin.setValue(gen["max_tokens"])
        self.min_tokens_spin.setValue(gen["min_tokens"])
        self.temperature_spin.setValue(gen["temperature"])
        self.top_k_spin.setValue(gen["top_k"])
        self.top_p_spin.setValue(gen["top_p"])
        self.rep_penalty_spin.setValue(gen["repetition_penalty"])
        self.stop_tokens_edit.setText(",".join(str(x) for x in gen.get("stop_token_ids", [])))
        self.stop_strings_edit.setText(",".join(gen.get("stop_strings", [])))

    # -- Training tab --
