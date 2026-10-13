"""Tkinter UI mixin (stdlib fallback). Part of the `uitk/` package split from `ui.py`."""
import json
import os
import queue
import threading
import time
import tkinter as tk
from tkinter import scrolledtext, messagebox, ttk, filedialog
from pathlib import Path

from tokenizer import BPETokenizer
from ai import TinyLM, train, generate, load_model, save_model, sync_vocab, train_parallel, count_params, Router, RouterTrainingExample
from registry import MODEL_REGISTRY, TASK_REGISTRY, TOKENIZER_MANAGER, get_tokenizer
from model_mgmt import ModelCatalog, ModelCreator, ModelLoader, MODEL_CATALOG
from config import Config
from autotrain import RatingCollector, FeedbackEntry, create_autotrain_system
from ui_base import (
    CONFIG_PATH,
    MODEL_PATH,
    PALETTE,
    PALETTE_LIGHT,
    PALETTE_DARK,
    TOKENIZER_PATH,
    get_available_models,
    load_config,
    apply_tk_theme,
    get_palette,
)
from ui_base import parse_bool as _parse_bool
from ui_base import write_config_yaml as _write_config_yaml
from uitk.const import cfg, VOCAB_SIZE, CTX, EMB, HIDDEN


class HistoryMixin:
    def _append_history(self, role, text):
        """Append a message to conversation history with role styling."""
        self.history.configure(state="normal")
        
        if role == "user":
            self.history.insert("end", "You: ", "user")
            self.history.insert("end", text + "\n\n")
            # Store user prompt for rating the next response
            self._last_user_prompt = text
        elif role == "assistant":
            self.history.insert("end", "AI: ", "assistant")
            self.history.insert("end", text + "\n\n")
            # Store for rating
            self._last_assistant_response = (getattr(self, '_last_user_prompt', ''), text, self.model_var.get())
            # Add rating buttons after the response
            self._add_rating_buttons(text)
        elif role == "system":
            self.history.insert("end", text + "\n\n", "system")
        elif role == "error":
            self.history.insert("end", text + "\n\n", "error")
        
        self.history.see("end")
        self.history.configure(state="disabled")
        
        # Also log to unified log viewer
        self._log_unified("system" if role != "error" else "error", text)

    def _add_rating_buttons(self, response_text):
        """Add rating buttons (1-5) after an assistant response."""
        # Create a frame for rating buttons
        rating_frame = tk.Frame(self.history)
        
        p = get_palette(self.current_theme)
        tk.Label(rating_frame, text="Rate: ", font=("Consolas", 8), foreground=p["muted"]).pack(side="left")
        
        for i in range(1, 6):
            if i <= 2:
                bg, fg = p["bg"], p["text"]
            elif i == 3:
                bg, fg = p["warning"], "white"
            else:
                bg, fg = p["success"], "white"
            btn = tk.Button(rating_frame, text=str(i), width=2, 
                          command=lambda r=i: self._on_rate_response(r),
                          font=("Consolas", 8),
                          bg=bg, fg=fg)
            btn.pack(side="left", padx=1)
        
        tk.Label(rating_frame, text="  ", font=("Consolas", 8)).pack(side="left")  # Spacer
        
        # Embed the frame in the text widget
        self.history.window_create("end", window=rating_frame)
        self.history.insert("end", "\n\n")  # Add spacing after rating buttons

    def _on_rate_response(self, rating):
        """Handle rating button click for the last assistant response."""
        if self._last_assistant_response is None:
            messagebox.showwarning("No response", "No response to rate.")
            return
        
        prompt, response, model_name = self._last_assistant_response
        
        try:
            entry = self.rating_collector.collect(
                prompt=prompt,
                response=response,
                model_name=model_name,
                rating=rating,
                metadata={"timestamp": time.time()}
            )
            self._log_unified("system", f"Feedback collected: {rating}/5 for {model_name}")
        except Exception as e:
            self._log_unified("error", f"Failed to record rating: {e}")

    def _log_train(self, message, is_error=False):
        """Append a timestamped line to the training monitor."""
        ts = time.strftime("%H:%M:%S")
        self.train_log.configure(state="normal")
        self.train_log.insert("end", f"[{ts}] ", "ts")
        self.train_log.insert("end", message + "\n", "err" if is_error else "")
        self.train_log.see("end")
        self.train_log.configure(state="disabled")
        
        # Also log to unified log viewer
        self._log_unified("error" if is_error else "training", message)

    def _log_unified(self, log_type, message):
        """Add entry to unified log viewer."""
        ts = time.strftime("%H:%M:%S")
        entry = (ts, log_type, message)
        self._all_log_entries.append(entry)
        
        # Keep only last 10000 entries to prevent memory issues
        if len(self._all_log_entries) > 10000:
            self._all_log_entries = self._all_log_entries[-10000:]
        
        # Apply current filters and display
        self._apply_log_filters()

    def _apply_log_filters(self, event=None):
        """Apply filters to unified log display."""
        if not hasattr(self, 'unified_log'):
            return
            
        search_text = self.log_search_var.get().lower() if hasattr(self, 'log_search_var') else ""
        
        # Determine which log types to show
        show_types = set()
        if hasattr(self, 'log_filter_vars'):
            for key, var in self.log_filter_vars.items():
                if var.get():
                    show_types.add(key)
        else:
            show_types = {"training", "generation", "system", "error"}
        
        self.unified_log.configure(state="normal")
        self.unified_log.delete("1.0", "end")
        
        for ts, log_type, message in self._all_log_entries:
            if log_type not in show_types:
                continue
            if search_text and search_text not in message.lower() and search_text not in ts.lower():
                continue
            
            self.unified_log.insert("end", f"[{ts}] ", "timestamp")
            self.unified_log.insert("end", f"[{log_type.upper()}] ", log_type)
            self.unified_log.insert("end", message + "\n", log_type)
        
        if self.log_autoscroll_var.get():
            self.unified_log.see("end")
        self.unified_log.configure(state="disabled")

    def _clear_unified_logs(self):
        """Clear all unified log entries."""
        self._all_log_entries.clear()
        self.unified_log.configure(state="normal")
        self.unified_log.delete("1.0", "end")
        self.unified_log.configure(state="disabled")

    def _export_logs(self):
        """Export unified logs to a text file."""
        filepath = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
            title="Export Logs"
        )
        if not filepath:
            return
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                for ts, log_type, message in self._all_log_entries:
                    f.write(f"[{ts}] [{log_type.upper()}] {message}\n")
            self._log_unified("system", f"Logs exported to {filepath}")
        except Exception as e:
            self._log_unified("error", f"Failed to export logs: {e}")
