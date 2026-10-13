"""
PySide6 (Qt6) primary GUI for ScratchpadLM.

This is the primary UI toolkit (roadmap 0a: "UI Toolkit Migration"). Tkinter
remains available as a stdlib-only fallback (``ui.py``); both toolkits share
the same colors (``ui_base.PALETTE``), helpers, and background-thread model.

Tabs: **Chat**, **Training**, **Model Info**, **Admin**, **Settings**, **Logs**.

Threading rules (same as the Tkinter app):
- Qt widgets are touched only on the main thread.
- Jobs run in daemon threads via ``ui_base.BackgroundWorker``.
- Worker results are delivered by posting on the queue; a ``QTimer`` drains
  the queue on the main thread every 100 ms.
"""

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

MONO = QFont("Monospace", 9)
MONO.setStyleHint(QFont.StyleHint.TypeWriter)


# ---------------------------------------------------------------------------
# Chat bubbles
# ---------------------------------------------------------------------------

class MessageBubble(QWidget):
    """Base chat message bubble (user / assistant / system / error)."""

    def __init__(self, role: str, text: str):
        super().__init__()
        self.role = role
        self.text_label = QLabel(text)
        self.text_label.setWordWrap(True)
        self.text_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self.text_label.setTextFormat(Qt.TextFormat.PlainText)
        self.text_label.setMaximumWidth(680)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 2, 0, 2)
        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)

        p = PALETTE
        if role == "user":
            self.text_label.setStyleSheet(
                f"background: {p['accent']}; color: white; padding: 8px 12px;"
                "border-radius: 10px; font-weight: bold;"
            )
            body.addStretch(1)
            body.addWidget(self.text_label)
        elif role == "assistant":
            self.text_label.setStyleSheet(
                f"background: {p['surface']}; color: {p['text']};"
                f"border: 1px solid {p['border']}; border-left: 4px solid {p['success']};"
                "padding: 8px 12px; border-radius: 10px;"
            )
            body.addWidget(self.text_label)
            body.addStretch(1)
        else:  # system / error
            color = p["error"] if role == "error" else p["muted"]
            style = "font-style: italic;" if role == "system" else "font-weight: bold;"
            self.text_label.setStyleSheet(f"color: {color}; padding: 2px 6px; {style}")
            body.addStretch(1)
            body.addWidget(self.text_label)
            body.addStretch(1)

        outer.addLayout(body)


class AssistantBubble(MessageBubble):
    """Assistant bubble with thumbs up/down feedback (rating 5 / 1)."""

    def __init__(self, text: str = ""):
        super().__init__("assistant", text)
        p = PALETTE

        self.feedback_row = QWidget()
        fb = QHBoxLayout(self.feedback_row)
        fb.setContentsMargins(14, 0, 0, 0)
        fb.setSpacing(6)

        rate_label = QLabel("Rate:")
        rate_label.setStyleSheet(f"color: {p['muted']}; font-size: 11px;")
        fb.addWidget(rate_label)

        self.thumbs_up = QPushButton("👍")
        self.thumbs_up.setToolTip("Good answer — feeds auto-training (rating 5/5)")
        self.thumbs_down = QPushButton("👎")
        self.thumbs_down.setToolTip("Bad answer — flags for review (rating 1/5)")
        for b in (self.thumbs_up, self.thumbs_down):
            b.setFixedWidth(34)
            fb.addWidget(b)

        self.rated_label = QLabel("")
        self.rated_label.setStyleSheet(f"color: {p['success']}; font-size: 11px;")
        fb.addWidget(self.rated_label)
        fb.addStretch(1)

        # Insert the feedback row under the bubble text
        self.layout().addWidget(self.feedback_row)
        self.feedback_row.hide()

    def show_feedback(self):
        self.feedback_row.show()

    def set_rated(self, rating: int):
        self.thumbs_up.setEnabled(False)
        self.thumbs_down.setEnabled(False)
        mark = "👍" if rating >= 4 else "👎"
        self.rated_label.setText(f"Rated {rating}/5 {mark}")


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

class MainWindow(QMainWindow):
    """ScratchpadLM main window (PySide6)."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("ScratchpadLM (TinyLM)")
        self.resize(1180, 780)

        self.cfg = load_config(CONFIG_PATH)
        self.tok = TOKENIZER_MANAGER.tokenizer
        if os.path.exists(TOKENIZER_PATH):
            TOKENIZER_MANAGER.load(TOKENIZER_PATH)

        # Apply theme from config
        theme = self.cfg.get("ui", "theme", "system")
        self.current_theme = theme
        self.setStyleSheet(qss_theme(theme))

        self.model = self._load_model()
        self.router = self._load_router()

        self.rating_collector = RatingCollector()
        self.worker = BackgroundWorker()
        self.busy = False
        self._model_name = "TinyLM"
        self._initialized = False  # suppress signals while populating widgets

        # Chat sessions: [{"title": str, "messages": [{"role","text","rating",...}]}]
        self._sessions = []
        self._current_session = -1
        self._stream_bubble = None      # AssistantBubble receiving streamed tokens
        self._last_assistant_response = None  # (prompt, response, model_name)

        self._all_log_entries = []  # (timestamp, log_type, message)
        self._catalog_names = []
        self._settings_entries = {}  # (section, key) -> (QLineEdit, cast)

        self._build_ui()

        self._poll_timer = QTimer(self)
        self._poll_timer.setInterval(100)
        self._poll_timer.timeout.connect(self._poll_worker)
        self._poll_timer.start()

        self._new_session(initial=True)
        self.refresh_models()
        self.refresh_catalog_list()
        self.update_model_inspector()
        self._initialized = True

    # ---- backend setup (shared logic with the Tkinter app) -----------------

    def _load_router(self):
        """Load or create the Router model."""
        try:
            router = load_model("router.tlm")
            if len(router.C) == len(self.tok.vocab):
                return router
        except Exception:
            pass
        config = RouterConfig(
            vocab_size=len(self.tok.vocab),
            ctx=self.cfg["model"]["ctx"],
            emb=self.cfg["model"]["emb"],
            hidden=self.cfg["model"]["hidden"],
            n_layers=2,
            d_ff=512,
        )
        return Router(config, seed=42)

    def _load_model(self):
        """Load the saved model if it matches the tokenizer, else create one."""
        try:
            model = load_model(MODEL_PATH)
            if len(model.C) == len(self.tok.vocab):
                return model
        except Exception:
            pass
        m = self.cfg["model"]
        return TinyLM(vocab_size=len(self.tok.vocab), ctx=m["ctx"], emb=m["emb"], hidden=m["hidden"])

    def _load_model_job(self, model_name):
        """Background job: build/load a model by name."""
        if model_name == "TinyLM":
            m = self.cfg["model"]
            return TinyLM(vocab_size=len(self.tok.vocab), ctx=m["ctx"], emb=m["emb"], hidden=m["hidden"])
        try:
            factory = MODEL_REGISTRY.get(model_name)
            return factory()
        except KeyError:
            try:
                loader = ModelLoader(MODEL_CATALOG, tokenizer_manager=TOKENIZER_MANAGER)
                return loader.load_model(model_name)
            except Exception as e:
                raise ValueError(f"Model '{model_name}' not found in registry or catalog: {e}")

    # ---- UI construction ---------------------------------------------------

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        outer = QVBoxLayout(central)
        outer.setContentsMargins(8, 8, 8, 4)

        self.tabs = QTabWidget()
        outer.addWidget(self.tabs)
        self.tabs.addTab(self._build_chat_tab(), "Chat")
        self.tabs.addTab(self._build_training_tab(), "Training")
        self.tabs.addTab(self._build_model_info_tab(), "Model Info")
        self.tabs.addTab(self._build_admin_tab(), "Admin")
        self.tabs.addTab(self._build_settings_tab(), "Settings")
        self.tabs.addTab(self._build_logs_tab(), "Logs")

        self.status_label = QLabel("Ready")
        sb = QStatusBar()
        sb.addWidget(self.status_label, 1)
        self.setStatusBar(sb)

    # -- Chat tab (sidebar list | chat area | input + send) --

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
        self.vocab_label = QLabel(f"Vocab: {len(self.tok.vocab)}")
        self.vocab_label.setStyleSheet(f"color: {p['muted']};")
        mrow.addWidget(self.vocab_label)
        mrow.addStretch(1)
        c.addLayout(mrow)

        # Chat area: scrollable column of bubbles
        self.chat_scroll = QScrollArea()
        self.chat_scroll.setWidgetResizable(True)
        self.chat_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._chat_host = QWidget()
        self._chat_vbox = QVBoxLayout(self._chat_host)
        self._chat_vbox.setContentsMargins(6, 6, 6, 6)
        self._chat_vbox.addStretch(1)
        self.chat_scroll.setWidget(self._chat_host)
        c.addWidget(self.chat_scroll, 1)

        # Quick generation settings
        gen = QGroupBox("Generation Settings")
        g = QHBoxLayout(gen)
        self.n_tokens_spin = QSpinBox()
        self.n_tokens_spin.setRange(1, 10000)
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
        for label, widget in (
            ("Tokens", self.n_tokens_spin),
            ("Temp", self.temperature_spin),
            ("Top-k", self.top_k_spin),
            ("Top-p", self.top_p_spin),
            ("Rep. penalty", self.rep_penalty_spin),
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
        self.n_tokens_spin.setValue(gen["n_tokens"])
        self.temperature_spin.setValue(gen["temperature"])
        self.top_k_spin.setValue(gen["top_k"])
        self.top_p_spin.setValue(gen["top_p"])
        self.rep_penalty_spin.setValue(gen["repetition_penalty"])

    # -- Training tab --

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

    def _build_settings_tab(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        body = QWidget()
        grid = QGridLayout(body)
        grid.setColumnStretch(0, 0)
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(2, 0)
        grid.setColumnStretch(3, 1)

        settings = [
            ("generation", "n_tokens", "Generation tokens", int),
            ("generation", "temperature", "Temperature", float),
            ("generation", "top_k", "Top-k", int),
            ("generation", "top_p", "Top-p", float),
            ("generation", "repetition_penalty", "Repetition penalty", float),
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

    def _new_session(self, initial: bool = False, switch: bool = True):
        self._sessions.append({"title": f"Chat {len(self._sessions) + 1}", "messages": []})
        idx = len(self._sessions) - 1
        self.chat_list.addItem(self._sessions[idx]["title"])
        if switch:
            self._current_session = -1  # force reload
            self.chat_list.setCurrentRow(idx)

    def _on_session_selected(self, row: int):
        if row < 0 or row == self._current_session or not self._initialized:
            if row >= 0 and not self._initialized:
                self._current_session = row
            return
        self._current_session = row
        self._rebuild_chat_area()

    def _rebuild_chat_area(self):
        """Re-render the current session's messages as bubbles."""
        while self._chat_vbox.count():
            item = self._chat_vbox.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        self._stream_bubble = None
        session = self._sessions[self._current_session]
        for msg in session["messages"]:
            self._add_bubble(msg, save=False)
        self._chat_vbox.addStretch(1)
        self._scroll_chat_to_bottom()

    def _append_message(self, role: str, text: str, **extra) -> dict:
        """Store a message in the current session and render it."""
        msg = {"role": role, "text": text, "rating": None}
        msg.update(extra)
        session = self._sessions[self._current_session]
        session["messages"].append(msg)
        # Rename "Chat N" sessions after the first user prompt
        if role == "user" and session["title"].startswith("Chat "):
            session["title"] = text[:32] + ("…" if len(text) > 32 else "")
            self.chat_list.item(self._current_session).setText(session["title"])
        self._add_bubble(msg, save=True)
        self._scroll_chat_to_bottom()
        return msg

    def _add_bubble(self, msg: dict, save: bool):
        if msg["role"] == "assistant":
            bubble = AssistantBubble(msg["text"])
            prompt = msg.get("prompt")
            if prompt:
                bubble.show_feedback()
                bubble.thumbs_up.clicked.connect(
                    lambda checked=False, m=msg, b=bubble: self._rate_response(m, b, 5)
                )
                bubble.thumbs_down.clicked.connect(
                    lambda checked=False, m=msg, b=bubble: self._rate_response(m, b, 1)
                )
                if msg.get("rating"):
                    bubble.set_rated(msg["rating"])
        else:
            bubble = MessageBubble(msg["role"], msg["text"])
        # Keep the trailing stretch last
        self._chat_vbox.insertWidget(self._chat_vbox.count() - 1, bubble)
        self._last_bubble = bubble
        return bubble

    def _scroll_chat_to_bottom(self):
        sb = self.chat_scroll.verticalScrollBar()
        QTimer.singleShot(0, lambda: sb.setValue(sb.maximum()))

    def append_history(self, role: str, text: str):
        """Public helper mirroring the Tkinter app's history API."""
        extra = {}
        if role == "assistant":
            prompt = getattr(self, "_last_user_prompt", "")
            extra = {"prompt": prompt, "model": self._model_name}
        self._append_message(role, text, **extra)
        self.log_unified("error" if role == "error" else "system", text)

    # -- feedback (thumbs up/down → RatingCollector) --

    def _rate_response(self, msg: dict, bubble: AssistantBubble, rating: int):
        if msg.get("rating") is not None:
            return
        model_name = msg.get("model", self._model_name)
        try:
            self.rating_collector.collect(
                prompt=msg.get("prompt", ""),
                response=msg["text"],
                model_name=model_name,
                rating=rating,
                metadata={"timestamp": time.time(), "source": "qt_ui"},
            )
            msg["rating"] = rating
            bubble.set_rated(rating)
            self.statusBar().showMessage(f"Rated {rating}/5 for '{model_name}'", 5000)
            self.log_unified("system", f"Feedback collected: {rating}/5 for {model_name}")
        except Exception as exc:
            self.show_error("Rating failed", str(exc))

    # ---- background worker plumbing ----------------------------------------

    def _poll_worker(self):
        for kind, label, payload in self.worker.drain():
            if kind == "progress":
                if label in ("Training", "Router Training"):
                    done, total, loss = payload
                    self.train_progress.setMaximum(max(total, 1))
                    self.train_progress.setValue(done)
                    self.progress_label.setText(f"Step {done}/{total}  loss {loss:.3f}")
            elif kind == "log":
                self.log_unified(label, payload)
            elif kind == "stream_token":
                # Posted as ("stream_token", token_str, (idx, total, tps)):
                # the token travels in the label slot, stats in the payload
                token_str = label
                token_idx, total_tokens, tps = payload
                self._handle_stream_token(token_str, token_idx, total_tokens, tps)
            else:
                self._handle_result(kind, label, payload)

    def _handle_stream_token(self, token_str, token_idx, total_tokens, tps):
        if self._stream_bubble is None:
            self._append_message("assistant", "")
            self._stream_bubble = getattr(self, "_last_bubble", None)
        if self._stream_bubble is not None:
            label = self._stream_bubble.text_label
            label.setText(label.text() + token_str)
            # Keep the stored message in sync so session switches keep the text
            session = self._sessions[self._current_session]
            if session["messages"]:
                session["messages"][-1]["text"] = label.text()
        self.status_label.setText(
            f"Generating... {token_idx}/{total_tokens} tokens ({tps:.1f} tok/s)"
        )
        self._scroll_chat_to_bottom()

    def set_busy(self, busy: bool, message: str = ""):
        self.busy = busy
        state = not busy
        for w in (
            self.train_btn, self.train_router_btn, self.send_btn, self.preview_btn,
            self.create_btn, self.delete_btn, self.new_chat_btn, self.refresh_models_btn,
            self.refresh_catalog_btn, self.model_combo,
        ):
            w.setEnabled(state)
        self.prompt_input.setEnabled(state)
        self.status_label.setText(message or ("Working..." if busy else "Ready"))

    # ---- result handling ----------------------------------------------------

    def _handle_result(self, kind: str, label: str, payload):
        if kind == "error":
            self._stream_bubble = None
            self.set_busy(False, "Error")
            self.append_history("error", f"Error: {payload}")
            if label == "Training":
                self.progress_label.setText("Failed")
            elif label == "Extending vocab":
                self._log_train(f"Vocab extension failed: {payload}", is_error=True)
            self.show_error(f"{label} failed", str(payload))
            return

        if label == "Training":
            prompt, steps = payload
            save_model(self.model, MODEL_PATH)
            self.vocab_label.setText(f"Vocab: {len(self.tok.vocab)}")
            self.append_history(
                "system",
                f"[Trained on {len(prompt)} chars, {steps} steps, model saved. "
                f"Vocab: {len(self.tok.vocab)}]",
            )
            self._log_train(
                f"Training done: {steps} steps on {len(prompt)} chars, "
                f"model saved, vocab={len(self.tok.vocab)}"
            )
            self.train_progress.setMaximum(max(steps, 1))
            self.train_progress.setValue(steps)
            self.progress_label.setText(f"Done ({steps} steps)")
            self.set_busy(False, "Training done")
            self.update_model_inspector()
        elif label == "Training Router":
            steps = payload
            self._log_train(f"Router training done: {steps} steps")
            self.train_progress.setMaximum(max(steps, 1))
            self.train_progress.setValue(steps)
            self.progress_label.setText(f"Done ({steps} steps)")
            self.set_busy(False, "Router training done")
        elif label.startswith("Loading "):
            self.model = payload
            model_name = label.replace("Loading ", "")
            self.append_history(
                "system",
                f"[Loaded model: {model_name} "
                f"({len(self.model.C)} vocab, {self.model.ctx} ctx)]",
            )
            self.set_busy(False, "Ready")
            self.update_model_inspector()
        elif label == "Generating":
            prompt, text, routed_model = payload
            self._last_assistant_response = (prompt, text, routed_model)
            if self._stream_bubble is not None:
                # Tokens were streamed live: close the bubble, enable feedback
                session = self._sessions[self._current_session]
                if session["messages"]:
                    session["messages"][-1].update(
                        {"prompt": prompt, "model": routed_model, "text": text}
                    )
                self._rebuild_chat_area()
                self._stream_bubble = None
            else:
                self._append_message("assistant", text, prompt=prompt, model=routed_model)
            self.set_busy(False, "Ready")
            self.status_label.setText("Ready")
        elif label == "Generating (Preview)":
            prompt, result = payload
            self._last_assistant_response = (prompt, result["text"], self._model_name)
            self._append_message(
                "assistant", result["text"], prompt=prompt, model=self._model_name
            )
            self.set_busy(False, "Ready")
            self.status_label.setText(f"Ready ({result['tokens_per_sec']:.1f} tok/s)")
        elif label == "Creating model":
            spec = payload
            self.append_history(
                "system",
                f"[Created model '{spec.name}' ({spec.param_count:,} params, "
                f"domains: {', '.join(spec.domain_keywords)})]",
            )
            self.refresh_catalog_list()
            self.refresh_models()
            self.set_busy(False, "Model created")
        elif label == "Deleting model":
            name, removed = payload
            if removed:
                self.append_history("system", f"[Deleted model '{name}' from catalog]")
            else:
                self.append_history("system", f"[Model '{name}' was not in catalog]")
            self.refresh_catalog_list()
            self.refresh_models()
            self.set_busy(False, "Model deleted")
        elif label == "Extending vocab":
            target_vocab, min_freq = payload
            self.vocab_label.setText(f"Vocab: {len(self.tok.vocab)}")
            self.tok_vocab_label.setText(str(len(self.tok.vocab)))
            self.tok_merges_label.setText(str(len(self.tok.merges)))
            self.append_history(
                "system",
                f"[Extended vocab to {len(self.tok.vocab)} tokens "
                f"(target={target_vocab}, min_freq={min_freq}), model saved]",
            )
            self._log_train(
                f"Vocab extended to {len(self.tok.vocab)} tokens, model saved"
            )
            self.set_busy(False, "Vocab extended")
            self.refresh_tokenizer_patterns()
            self.update_model_inspector()
        else:
            self.set_busy(False, "Ready")

    # ---- logging -------------------------------------------------------------

    def log_train(self, message: str, is_error: bool = False):
        self._log_train(message, is_error)

    def _log_train(self, message: str, is_error: bool = False):
        ts = time.strftime("%H:%M:%S")
        color = PALETTE["error"] if is_error else PALETTE["text"]
        self.train_log.appendHtml(
            f'<span style="color:{PALETTE["muted"]}">[{ts}]</span> '
            f'<span style="color:{color}">{self._escape(message)}</span>'
        )
        self.log_unified("error" if is_error else "training", message)

    def log_unified(self, log_type: str, message: str):
        ts = time.strftime("%H:%M:%S")
        self._all_log_entries.append((ts, log_type, message))
        if len(self._all_log_entries) > 10000:
            self._all_log_entries = self._all_log_entries[-10000:]
        self._apply_log_filters()

    def _apply_log_filters(self, event=None):
        search = self.log_search_edit.text().strip().lower()
        show = {k for k, cb in self.log_filter_boxes.items() if cb.isChecked()}
        html = []
        colors = {
            "training": PALETTE["accent"],
            "generation": PALETTE["success"],
            "system": PALETTE["muted"],
            "error": PALETTE["error"],
        }
        for ts, log_type, message in self._all_log_entries:
            if log_type not in show:
                continue
            if search and search not in message.lower() and search not in ts.lower():
                continue
            color = colors.get(log_type, PALETTE["text"])
            kind = log_type.upper()
            html.append(
                f'<span style="color:#95A5A6">[{ts}]</span> '
                f'<b><span style="color:{color}">[{kind}]</span></b> '
                f'<span style="color:{color}">{self._escape(message)}</span>'
            )
        self.unified_log.setHtml("<br>".join(html))
        if self.log_autoscroll_cb.isChecked():
            self.unified_log.moveCursor(QTextCursor.MoveOperation.End)

    def _clear_unified_logs(self):
        self._all_log_entries.clear()
        self.unified_log.clear()

    def _export_logs(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Logs", "logs.txt", "Text files (*.txt);;All files (*)"
        )
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as f:
                for ts, log_type, message in self._all_log_entries:
                    f.write(f"[{ts}] [{log_type.upper()}] {message}\n")
            self.log_unified("system", f"Logs exported to {path}")
        except Exception as exc:
            self.log_unified("error", f"Failed to export logs: {exc}")

    @staticmethod
    def _escape(text: str) -> str:
        return (
            str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        )

    # ---- refresh helpers ------------------------------------------------------

    def refresh_models(self):
        models = get_available_models()
        self.model_combo.blockSignals(True)
        self.model_combo.clear()
        self.model_combo.addItems(models)
        if self._model_name not in models:
            self._model_name = models[0] if models else "TinyLM"
        self.model_combo.setCurrentText(self._model_name)
        self.model_combo.blockSignals(False)
        self.target_label.setText(self._model_name)
        self.append_history("system", f"[Refreshed model list: {', '.join(models)}]")

    def _on_model_change(self, name: str):
        if not self._initialized or not name or self.busy:
            return
        self._model_name = name
        self.target_label.setText(name)
        self.append_history("system", f"[Switching to model: {name}...]")
        self.worker.submit(f"Loading {name}", self._load_model_job, name)

    def refresh_catalog_list(self):
        self.admin_list.clear()
        self._catalog_names = []
        for spec in MODEL_CATALOG.list():
            domains = ", ".join(spec.domain_keywords) or "-"
            self.admin_list.addItem(
                f"{spec.name}  |  {spec.param_count:,} params  |  {domains}"
            )
            self._catalog_names.append(spec.name)

    def update_model_inspector(self):
        if self.model is None:
            for lbl in (
                self.inspector_model_name, self.inspector_arch, self.inspector_params,
                self.inspector_memory, self.inspector_vocab, self.inspector_ctx,
                self.inspector_emb, self.inspector_hidden,
            ):
                lbl.setText("None" if lbl is self.inspector_model_name else "0")
            self.tensor_detail_btn.setEnabled(False)
            return

        config = self.model.get_config()
        arch = config.get("arch", "TinyLM")
        vocab_size = config.get("vocab_size", len(self.model.C))
        ctx = config.get("ctx", getattr(self.model, "ctx", 0))
        emb = config.get("emb", len(self.model.C[0]) if self.model.C else 0)
        b1 = getattr(self.model, "b1", None)
        hidden = config.get("hidden", len(b1) if b1 else 0)

        try:
            param_count = count_params(self.model)
        except Exception:
            param_count = 0
        memory_bytes = param_count * 0.5
        memory_str = (
            f"{memory_bytes / (1024 * 1024):.2f} MB"
            if memory_bytes >= 1024 * 1024
            else f"{memory_bytes / 1024:.1f} KB"
        )

        self.inspector_model_name.setText(self._model_name)
        self.inspector_arch.setText(arch)
        self.inspector_params.setText(f"{param_count:,}")
        self.inspector_memory.setText(memory_str)
        self.inspector_vocab.setText(str(vocab_size))
        self.inspector_ctx.setText(str(ctx))
        self.inspector_emb.setText(str(emb))
        self.inspector_hidden.setText(str(hidden))
        self.tensor_detail_btn.setEnabled(True)

    def refresh_tokenizer_patterns(self):
        try:
            patterns = self.tok.learned_patterns(min_len=2)
            display = patterns[:100]
            text = "\n".join(display) if display else "(no patterns learned yet)"
            if len(patterns) > 100:
                text += f"\n... ({len(patterns)} total patterns)"
            self.tok_patterns_text.setPlainText(text)
            self.tok_vocab_label.setText(str(len(self.tok.vocab)))
            self.tok_merges_label.setText(str(len(self.tok.merges)))
            self.vocab_label.setText(f"Vocab: {len(self.tok.vocab)}")
        except Exception as exc:
            self.tok_patterns_text.setPlainText(f"Error: {exc}")

    # ---- dialogs ---------------------------------------------------------------

    def show_error(self, title: str, message: str):
        QMessageBox.critical(self, title, message)

    def show_warning(self, title: str, message: str):
        QMessageBox.warning(self, title, message)

    def ask_yes_no(self, title: str, message: str) -> bool:
        return (
            QMessageBox.question(self, title, message)
            == QMessageBox.StandardButton.Yes
        )

    # ---- actions -----------------------------------------------------------------

    def on_send_prompt(self):
        if self.busy:
            return
        prompt = self.prompt_input.text().strip()
        if not prompt:
            return
        self.prompt_input.clear()
        self._last_user_prompt = prompt
        self._append_message("user", prompt)
        self._stream_bubble = None
        self.worker.submit(
            "Generating",
            self._generate_job,
            prompt,
            self.n_tokens_spin.value(),
            self.temperature_spin.value(),
            self.top_k_spin.value(),
            self.top_p_spin.value(),
            self.rep_penalty_spin.value(),
        )
        self.set_busy(True, "Generating...")

    def _generate_job(self, prompt, n_tokens, temperature, top_k, top_p, repetition_penalty):
        model, model_name = self._route_prompt(prompt)

        def stream_callback(token_str, token_idx, total_tokens, tps):
            self.worker.post("stream_token", token_str, (token_idx, total_tokens, tps))

        example = self._collect_routing_data(prompt, model_name, "")
        text = generate(
            model, self.tok, prompt,
            n_tokens=n_tokens, temperature=temperature,
            top_k=top_k, top_p=top_p, repetition_penalty=repetition_penalty,
            stream_callback=stream_callback,
        )
        if example:
            example.subtasks[0]["response"] = text
            self._store_routing_example(example)

        self.worker.post("log", "generation", f"Prompt: {prompt[:100]}{'...' if len(prompt) > 100 else ''}")
        self.worker.post("log", "generation", f"Routed to: {model_name}")
        self.worker.post("log", "generation", f"Response: {text[:200]}{'...' if len(text) > 200 else ''}")
        return prompt, text, model_name

    def on_generate_preview(self):
        if self.busy:
            return
        prompt = self.prompt_input.text().strip()
        if not prompt:
            return
        self.prompt_input.clear()
        self._last_user_prompt = prompt
        self._append_message("user", prompt)
        self.worker.submit(
            "Generating (Preview)",
            self._generate_preview_job,
            prompt,
            self.n_tokens_spin.value(),
            self.temperature_spin.value(),
            self.top_k_spin.value(),
            self.top_p_spin.value(),
            self.rep_penalty_spin.value(),
        )
        self.set_busy(True, "Generating (Preview)...")

    def _generate_preview_job(self, prompt, n_tokens, temperature, top_k, top_p, repetition_penalty):
        result = generate_with_preview(
            self.model, self.tok, prompt,
            n_tokens=n_tokens, temperature=temperature,
            top_k=top_k, top_p=top_p, repetition_penalty=repetition_penalty,
        )
        return prompt, result

    def _route_prompt(self, prompt):
        """Route the prompt with the Router model (worker thread)."""
        current_name = self._model_name
        try:
            routes = self.router.route_multi(prompt, self.tok, MODEL_CATALOG, max_subtasks=3)
            if not routes:
                return self.model, current_name
            model_name = routes[0]["model"]
            if model_name != current_name:
                loader = ModelLoader(MODEL_CATALOG, tokenizer_manager=TOKENIZER_MANAGER)
                return loader.load_model(model_name), model_name
            return self.model, current_name
        except Exception as e:
            self.worker.post("log", "error", f"Routing failed: {e}, falling back to current model")
            return self.model, current_name

    def _collect_routing_data(self, prompt, model_name, response):
        try:
            return RouterTrainingExample(prompt=prompt, subtasks=[{"model": model_name, "prompt": prompt}])
        except Exception:
            return None

    def _store_routing_example(self, example):
        try:
            with open("router_training_data.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps({"prompt": example.prompt, "subtasks": example.subtasks}) + "\n")
        except Exception:
            pass

    def on_train(self):
        if self.busy:
            return
        prompt = self.train_text.toPlainText().strip()
        if not prompt or prompt == "Enter training text here...":
            self.show_warning("No input", "Enter some training text in the Training Data area.")
            return
        steps = self.steps_spin.value()
        self._log_train(f"Training started: {len(prompt)} chars, {steps} steps, model={self._model_name}")
        self.train_progress.setMaximum(steps)
        self.train_progress.setValue(0)
        self.progress_label.setText(f"Step 0/{steps}")
        self.worker.submit("Training", self._train_job, prompt, steps)
        self.set_busy(True, "Training...")

    def _train_job(self, prompt, steps):
        tokenizer_cfg = self.cfg["tokenizer"]
        training_cfg = self.cfg["training"]

        self.tok.extend(
            prompt,
            vocab_size=tokenizer_cfg["vocab_size"],
            min_frequency=tokenizer_cfg["min_frequency"],
        )
        self.tok.save(TOKENIZER_PATH)
        sync_vocab(self.model, len(self.tok.vocab))

        ids = self.tok.encode(prompt)
        if len(ids) <= self.model.ctx:
            raise ValueError(
                f"The input has {len(ids)} tokens, but training needs more than {self.model.ctx}."
            )

        def on_progress(done, total, loss):
            self.worker.post("progress", "Training", (done, total, loss))

        train_parallel(
            self.model, ids, steps=steps, lr=training_cfg["lr"],
            batch_size=training_cfg["batch_size"], n_negatives=training_cfg["n_negatives"],
            workers=training_cfg["workers"], progress_cb=on_progress,
        )
        return prompt, steps

    def on_train_router(self):
        if self.busy:
            return
        if not os.path.exists("router_training_data.jsonl"):
            self.show_warning(
                "No data",
                "No router training data found. Use the chat to generate routing data first.",
            )
            return
        steps = self.steps_spin.value()
        self._log_train(f"Router training started: {steps} steps")
        self.train_progress.setMaximum(steps)
        self.train_progress.setValue(0)
        self.progress_label.setText(f"Step 0/{steps}")
        self.worker.submit("Training Router", self._train_router_job, steps)
        self.set_busy(True, "Training Router...")

    def _train_router_job(self, steps):
        examples = []
        with open("router_training_data.jsonl", "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                examples.append(RouterTrainingExample(prompt=data["prompt"], subtasks=data["subtasks"]))
        if not examples:
            raise ValueError("No valid training examples")

        def on_progress(done, total, loss, val_loss=None, lr=None):
            self.worker.post("progress", "Router Training", (done, total, loss))

        training_cfg = self.cfg["training"]
        train_router(
            self.router, self.tok, examples,
            steps=steps, lr=training_cfg["lr"],
            batch_size=training_cfg["batch_size"],
            n_negatives=training_cfg["n_negatives"],
            progress_cb=on_progress,
        )
        save_model(self.router, "router.tlm")
        return steps

    # ---- admin actions -------------------------------------------------------

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
        self.append_history("system", f"[Creating model '{name}' (domains: {', '.join(domains)})...]")
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

    def _on_theme_change(self, theme: str):
        """Handle theme change from combobox."""
        if theme != self.current_theme:
            self.current_theme = theme
            self.setStyleSheet(qss_theme(theme))
            # Update bubble colors - they'll use the new palette on next render
            p = get_palette(theme)
            # Update status bar
            self.status_label.setStyleSheet(f"color: {p['text']};")

    # ---- config editor actions --------------------------------------------------

    def on_load_config_editor(self):
        """Load current config into the YAML editor."""
        import yaml
        try:
            data = self.cfg.to_dict()
            # Convert to YAML
            yaml_text = yaml.dump(data, default_flow_style=False, sort_keys=False)
            self.config_editor.setPlainText(yaml_text)
        except Exception as e:
            self.show_error("Load Config", f"Failed to load config: {e}")

    def on_validate_config_editor(self):
        """Validate the YAML in the editor."""
        from config import Config, ConfigError
        text = self.config_editor.toPlainText()
        try:
            import yaml
            data = yaml.safe_load(text)
            Config.from_dict(data)
            self.show_error("Validation", "Config is valid!")
        except ConfigError as e:
            self.show_error("Validation Error", str(e))
        except Exception as e:
            self.show_error("YAML Error", f"Invalid YAML: {e}")

    def on_save_config_editor(self):
        """Save config from the YAML editor."""
        from config import Config, ConfigError
        import yaml
        text = self.config_editor.toPlainText()
        try:
            data = yaml.safe_load(text)
            cfg = Config.from_dict(data)
            write_config_yaml(CONFIG_PATH, data)
            self.cfg = cfg
            # Refresh settings entries
            for (section, key), (edit, cast) in self._settings_entries.items():
                if section in data and key in data[section]:
                    val = data[section][key]
                    if isinstance(edit, QComboBox):
                        edit.setCurrentText(str(val))
                    else:
                        edit.setText(str(val))
            self.append_history("system", f"[Config saved from editor to {CONFIG_PATH}]")
        except ConfigError as e:
            self.show_error("Validation Error", str(e))
        except Exception as e:
            self.show_error("Save Error", f"Failed to save config: {e}")

    def on_preset_changed(self, preset_name: str):
        """Apply a preset configuration."""
        presets = {
            "Default": {},
            "Fast Training": {
                "training": {"steps": 100, "lr": 0.1, "batch_size": 32, "workers": 4}
            },
            "High Quality": {
                "training": {"steps": 2000, "lr": 0.01, "batch_size": 16, "n_negatives": 512, "val_split": 0.2}
            },
            "Low Memory": {
                "model": {"emb": 32, "hidden": 64, "ctx": 64},
                "training": {"batch_size": 8, "n_negatives": 128}
            },
        }
        if preset_name in presets and preset_name != "Custom...":
            preset = presets[preset_name]
            # Merge with current config
            data = self.cfg.to_dict()
            for section, values in preset.items():
                if section not in data:
                    data[section] = {}
                data[section].update(values)
            # Update editor
            import yaml
            yaml_text = yaml.dump(data, default_flow_style=False, sort_keys=False)
            self.config_editor.setPlainText(yaml_text)

    def on_save_preset(self):
        """Save current editor content as a custom preset."""
        # For now, just show the YAML in a dialog
        text = self.config_editor.toPlainText()
        dialog = QDialog(self)
        dialog.setWindowTitle("Save Preset")
        dialog.resize(500, 400)
        v = QVBoxLayout(dialog)
        v.addWidget(QLabel("Enter preset name:"))
        name_edit = QLineEdit()
        v.addWidget(name_edit)
        btn_row = QHBoxLayout()
        save_btn = QPushButton("Save")
        cancel_btn = QPushButton("Cancel")
        btn_row.addWidget(save_btn)
        btn_row.addWidget(cancel_btn)
        v.addLayout(btn_row)
        save_btn.clicked.connect(lambda: self._save_custom_preset(name_edit.text(), text, dialog))
        cancel_btn.clicked.connect(dialog.reject)
        dialog.exec()

    def _save_custom_preset(self, name: str, yaml_text: str, dialog: QDialog):
        """Save a custom preset."""
        if not name.strip():
            self.show_warning("Invalid Name", "Preset name cannot be empty.")
            return
        # In a full implementation, we'd save to a presets file
        self.show_error("Preset Saved", f"Preset '{name}' saved (in-memory only in this demo).")
        dialog.accept()

    # ---- settings actions ------------------------------------------------------

    def on_save_config(self):
        """Validate Settings entries and save to config.yaml (same rules as Tk)."""
        new_data = self.cfg.to_dict()
        for (section, key), (edit, cast) in self._settings_entries.items():
            raw = edit.text().strip()
            try:
                value = cast(raw)
            except ValueError:
                self.show_warning("Invalid value", f"{section}.{key}: '{raw}' is not a valid {cast.__name__}.")
                return
            ZERO_OK = {"top_k", "warmup_steps", "eval_every", "checkpoint_every"}
            if isinstance(value, int) and not isinstance(value, bool):
                if key in ZERO_OK and value < 0:
                    self.show_warning("Invalid value", f"{section}.{key} must be >= 0.")
                    return
                if key not in ZERO_OK and value < 1:
                    self.show_warning("Invalid value", f"{section}.{key} must be a positive whole number.")
                    return
            new_data[section][key] = value

        gen = new_data["generation"]
        checks = [
            (gen["temperature"] > 0, "generation.temperature must be greater than 0."),
            (gen["top_k"] >= 0, "generation.top_k must be >= 0."),
            (0.0 <= gen["top_p"] <= 1.0, "generation.top_p must be between 0.0 and 1.0."),
            (gen["repetition_penalty"] >= 1.0, "generation.repetition_penalty must be >= 1.0."),
        ]
        tr = new_data["training"]
        checks += [
            (tr["lr_scheduler"] in ("linear", "cosine_warmup", "constant"),
             "training.lr_scheduler must be linear, cosine_warmup or constant."),
            (tr["min_lr"] >= 0, "training.min_lr must be >= 0."),
            (tr["grad_clip"] >= 0, "training.grad_clip must be >= 0."),
            (0.0 <= tr["val_split"] < 1.0, "training.val_split must be in [0.0, 1.0)."),
            (tr["min_delta"] >= 0, "training.min_delta must be >= 0."),
            (isinstance(tr["early_stopping"], bool), "training.early_stopping must be true or false."),
            (0.0 < tr["auto_factor"] < 1.0, "training.auto_factor must be between 0.0 and 1.0."),
            (isinstance(tr["auto_adjust"], bool), "training.auto_adjust must be true or false."),
            (isinstance(tr["auto_clip_relax"], bool), "training.auto_clip_relax must be true or false."),
        ]
        for ok, msg in checks:
            if not ok:
                self.show_warning("Invalid value", msg)
                return

        try:
            write_config_yaml(CONFIG_PATH, new_data)
            self.cfg = Config.from_dict(new_data)
        except Exception as exc:
            self.show_error("Error", f"Failed to save config:\n{exc}")
            return

        # Sync quick-access widgets with the new defaults
        self.steps_spin.setValue(new_data["training"]["steps"])
        self._load_gen_settings()
        self.append_history("system", f"[Config saved to {CONFIG_PATH}]")

    # ---- tokenizer manager actions ----------------------------------------------

    def on_extend_vocab(self):
        if self.busy:
            return
        text = self.tok_extend_text.toPlainText().strip()
        if not text:
            self.show_warning("No input", "Enter some text to extend the vocabulary.")
            return
        try:
            target_vocab = int(self.tok_target_vocab_edit.text())
            min_freq = int(self.tok_min_freq_edit.text())
            if target_vocab < 256:
                raise ValueError("Target vocab size must be at least 256")
            if min_freq < 1:
                raise ValueError("Min frequency must be at least 1")
        except ValueError as e:
            self.show_warning("Invalid input", str(e))
            return
        self._log_train(f"Extending vocab: +{len(text)} chars, target={target_vocab}, min_freq={min_freq}")
        self.worker.submit("Extending vocab", self._extend_vocab_job, text, target_vocab, min_freq)
        self.set_busy(True, "Extending vocab...")

    def _extend_vocab_job(self, text, target_vocab, min_freq):
        TOKENIZER_MANAGER.extend(text, vocab_size=target_vocab, min_frequency=min_freq)
        TOKENIZER_MANAGER.save(TOKENIZER_PATH)
        sync_vocab(self.model, len(self.tok.vocab))
        save_model(self.model, MODEL_PATH)
        return target_vocab, min_freq

    # ---- tensor details ---------------------------------------------------------

    def on_show_tensor_details(self):
        if self.model is None:
            return
        config = self.model.get_config()
        tensors = self.model.get_tensors()

        dialog = QDialog(self)
        dialog.setWindowTitle(f"Tensor Details - {self._model_name}")
        dialog.resize(620, 520)
        v = QVBoxLayout(dialog)
        text = QTextEdit()
        text.setReadOnly(True)
        text.setFont(MONO)
        lines = [
            f"Model: {self._model_name}",
            f"Architecture: {config.get('arch', 'Unknown')}",
            f"Vocab Size: {config.get('vocab_size', 'N/A')}",
            f"Context: {config.get('ctx', 'N/A')}",
            f"Embedding Dim: {config.get('emb', 'N/A')}",
            f"Hidden Dim: {config.get('hidden', 'N/A')}",
            "",
        ]
        total_params = 0
        for name, rows in tensors.items():
            n_rows = len(rows)
            n_cols = len(rows[0]) if rows else 0
            tensor_params = n_rows * n_cols
            total_params += tensor_params
            lines.append(f"  {name}: {n_rows} x {n_cols} = {tensor_params:,} params")
        lines += ["", f"Total Parameters: {total_params:,}"]
        text.setPlainText("\n".join(lines))
        v.addWidget(text)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dialog.accept)
        v.addWidget(close_btn, alignment=Qt.AlignmentFlag.AlignHCenter)
        dialog.exec()


def run_qt_app() -> int:
    """Launch the PySide6 app (called by ui_base.run_gui)."""
    import sys

    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("ScratchpadLM")
    app.setStyleSheet(qss_theme())
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(run_qt_app())
