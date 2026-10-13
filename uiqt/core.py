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


class CoreMixin:
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ScratchpadLM (TinyLM)")
        self.resize(1180, 780)

        self.cfg = load_config(CONFIG_PATH)
        self.tok = TOKENIZER_MANAGER.tokenizer
        if os.path.exists(TOKENIZER_PATH):
            TOKENIZER_MANAGER.load(TOKENIZER_PATH)

        # Apply theme from config at application level
        theme = self.cfg.get("ui", "theme", "dark")
        self.current_theme = theme
        QApplication.instance().setStyleSheet(qss_theme(theme))

        self.model = self._load_model()
        self.router = self._load_router()

        self.rating_collector = RatingCollector()
        self.worker = BackgroundWorker()
        self.busy = False
        self._cancel_training = False  # Flag to signal training cancellation
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
        self.log_unified("system", f"Refreshed model list: {', '.join(models)}")

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
