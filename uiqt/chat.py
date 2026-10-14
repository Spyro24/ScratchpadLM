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
    normalize_cap,
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


class ChatMixin:
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
        current_palette = get_palette(self.current_theme)
        if msg["role"] == "assistant":
            bubble = AssistantBubble(msg["text"], palette=current_palette)
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
            bubble = MessageBubble(msg["role"], msg["text"], palette=current_palette)
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
        
        # Get generation settings from config (moved from Chat to Settings tab)
        gen = self.cfg["generation"]
        max_tokens = gen["max_tokens"]
        min_tokens = gen["min_tokens"]
        temperature = gen["temperature"]
        top_k = gen["top_k"]
        top_p = gen["top_p"]
        repetition_penalty = gen["repetition_penalty"]
        try:
            stop_token_ids = [int(x.strip()) for x in gen.get("stop_token_ids", [])]
        except (ValueError, TypeError):
            stop_token_ids = []
        stop_strings = gen.get("stop_strings", [])
        
        # Snapshot model-selection state on the main thread (Qt widgets must
        # only be touched on the main thread).
        use_selected = bool(self.use_selected_model_cb.isChecked())
        selected_name = self._model_name
        self.worker.submit(
            "Generating",
            self._generate_job,
            prompt,
            normalize_cap(max_tokens),
            min_tokens,
            temperature,
            top_k,
            top_p,
            repetition_penalty,
            stop_token_ids,
            stop_strings,
            use_selected,
            selected_name,
        )
        self.set_busy(True, "Generating...")

    def _generate_job(self, prompt, max_tokens, min_tokens, temperature, top_k, top_p, repetition_penalty, stop_token_ids, stop_strings, use_selected=False, selected_name=None):
        model, model_name = self._route_prompt(prompt, use_selected=use_selected, selected_name=selected_name)

        def stream_callback(token_str, token_idx, total_tokens, tps):
            self.worker.post("stream_token", token_str, (token_idx, total_tokens, tps))

        example = self._collect_routing_data(prompt, model_name, "")
        text = generate(
            model, self.tok, prompt,
            max_tokens=max_tokens, min_tokens=min_tokens, temperature=temperature,
            top_k=top_k, top_p=top_p, repetition_penalty=repetition_penalty,
            stop_token_ids=stop_token_ids, stop_strings=stop_strings,
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
        try:
            stop_token_ids = [int(x.strip()) for x in self.stop_tokens_edit.text().split(",") if x.strip()]
            stop_strings = [x.strip() for x in self.stop_strings_edit.text().split(",") if x.strip()]
        except ValueError:
            stop_token_ids = []
            stop_strings = []
        self.worker.submit(
            "Generating (Preview)",
            self._generate_preview_job,
            prompt,
            normalize_cap(self.max_tokens_spin.value()),
            self.min_tokens_spin.value(),
            self.temperature_spin.value(),
            self.top_k_spin.value(),
            self.top_p_spin.value(),
            self.rep_penalty_spin.value(),
            stop_token_ids,
            stop_strings,
        )
        self.set_busy(True, "Generating (Preview)...")

    def _generate_preview_job(self, prompt, max_tokens, min_tokens, temperature, top_k, top_p, repetition_penalty, stop_token_ids, stop_strings):
        result = generate_with_preview(
            self.model, self.tok, prompt,
            max_tokens=max_tokens, min_tokens=min_tokens, temperature=temperature,
            top_k=top_k, top_p=top_p, repetition_penalty=repetition_penalty,
            stop_token_ids=stop_token_ids, stop_strings=stop_strings,
        )
        return prompt, result

    def _route_prompt(self, prompt, use_selected=False, selected_name=None):
        """Route the prompt with the Router model (worker thread).

        When `use_selected` is True the manually selected model is used
        directly and the router is bypassed. `selected_name` must be
        snapshotted on the main thread by the caller.
        """
        current_name = selected_name or self._model_name

        # If user wants to use selected model, bypass router
        if use_selected:
            return self.model, current_name
        
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
