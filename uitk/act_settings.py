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
from tinylm import TinyLM, train, generate, load_model, save_model, sync_vocab, train_parallel, count_params, Router, RouterTrainingExample
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


class SettingsActionsMixin:
    def _on_theme_change(self, event=None):
        """Handle theme change from combobox."""
        new_theme = self.theme_var.get()
        if new_theme != self.current_theme:
            self.current_theme = new_theme
            apply_tk_theme(self.root, new_theme)
            # Refresh tag colors for history text
            p = get_palette(new_theme)
            self.history.tag_configure("user", foreground=p["accent"], font=("Consolas", 10, "bold"))
            self.history.tag_configure("assistant", foreground=p["success"], font=("Consolas", 10, "bold"))
            self.history.tag_configure("system", foreground=p["muted"], font=("Consolas", 9, "italic"))
            self.history.tag_configure("error", foreground=p["error"], font=("Consolas", 10, "bold"))
            self.train_log.tag_configure("ts", foreground=p["muted"])
            self.train_log.tag_configure("err", foreground=p["error"])
            self.unified_log.tag_configure("training", foreground=p["accent"])
            self.unified_log.tag_configure("generation", foreground=p["success"])
            self.unified_log.tag_configure("system", foreground=p["muted"], font=("Consolas", 9, "italic"))
            self.unified_log.tag_configure("error", foreground=p["error"], font=("Consolas", 9, "bold"))
            self.unified_log.tag_configure("timestamp", foreground="#95A5A6")
            self.status.configure(bg=p["bg"], fg=p["text"])

    # ---- config editor actions --------------------------------------------------

    def on_load_config_editor(self):
        """Load current config into the YAML editor."""
        import yaml
        try:
            data = self.cfg.to_dict()
            yaml_text = yaml.dump(data, default_flow_style=False, sort_keys=False)
            self.config_editor.delete("1.0", "end")
            self.config_editor.insert("1.0", yaml_text)
        except Exception as e:
            messagebox.showerror("Load Config", f"Failed to load config: {e}")

    def on_validate_config_editor(self):
        """Validate the YAML in the editor."""
        from config import Config, ConfigError
        import yaml
        text = self.config_editor.get("1.0", "end-1c")
        try:
            data = yaml.safe_load(text)
            Config.from_dict(data)
            messagebox.showinfo("Validation", "Config is valid!")
        except ConfigError as e:
            messagebox.showerror("Validation Error", str(e))
        except Exception as e:
            messagebox.showerror("YAML Error", f"Invalid YAML: {e}")

    def on_save_config_editor(self):
        """Save config from the YAML editor."""
        from config import Config, ConfigError
        import yaml
        text = self.config_editor.get("1.0", "end-1c")
        try:
            data = yaml.safe_load(text)
            cfg = Config.from_dict(data)
            _write_config_yaml(CONFIG_PATH, data)
            self.cfg = cfg
            # Refresh settings entries
            for (section, key), (entry, cast) in self._settings_entries.items():
                if section in data and key in data[section]:
                    val = data[section][key]
                    if isinstance(entry, ttk.Combobox):
                        entry.set(str(val))
                    else:
                        entry.delete(0, "end")
                        entry.insert(0, str(val))
            self._log_unified("system", f"Config saved from editor to {CONFIG_PATH}")
        except ConfigError as e:
            messagebox.showerror("Validation Error", str(e))
        except Exception as e:
            messagebox.showerror("Save Error", f"Failed to save config: {e}")

    def on_preset_changed(self, event=None):
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
        preset_name = self.preset_var.get()
        if preset_name in presets and preset_name != "Custom...":
            preset = presets[preset_name]
            data = self.cfg.to_dict()
            for section, values in preset.items():
                if section not in data:
                    data[section] = {}
                data[section].update(values)
            import yaml
            yaml_text = yaml.dump(data, default_flow_style=False, sort_keys=False)
            self.config_editor.delete("1.0", "end")
            self.config_editor.insert("1.0", yaml_text)

    def on_save_preset(self):
        """Save current editor content as a custom preset."""
        # For now, just show the YAML in a dialog
        text = self.config_editor.get("1.0", "end-1c")
        name = tk.simpledialog.askstring("Save Preset", "Enter preset name:")
        if name and name.strip():
            # In a full implementation, we'd save to a presets file
            messagebox.showinfo("Preset Saved", f"Preset '{name}' saved (in-memory only in this demo).")

    # ---- settings actions --------------------------------------------------

    def on_save_config(self):
        """Validate Settings tab entries and save to config.yaml."""
        new_data = self.cfg.to_dict()
        for (section, key), (entry, cast) in self._settings_entries.items():
            raw = entry.get().strip()
            try:
                if cast == str and key in ("stop_token_ids", "stop_strings"):
                    # Special handling for comma-separated lists
                    if key == "stop_token_ids":
                        value = [int(x.strip()) for x in raw.split(",") if x.strip()]
                    else:
                        value = [x.strip() for x in raw.split(",") if x.strip()]
                else:
                    value = cast(raw)
            except ValueError:
                messagebox.showwarning(
                    "Invalid value", f"{section}.{key}: '{raw}' is not a valid {cast.__name__}."
                )
                return
            # Validate integer fields: most must be positive; some allow zero
            ZERO_OK = {"top_k", "warmup_steps", "eval_every", "checkpoint_every", "min_tokens"}
            if isinstance(value, int) and not isinstance(value, bool):
                if key in ZERO_OK:
                    if value < 0:
                        messagebox.showwarning("Invalid value", f"{section}.{key} must be >= 0.")
                        return
                elif value < 1:
                    messagebox.showwarning("Invalid value", f"{section}.{key} must be a positive whole number.")
                    return
            new_data[section][key] = value
        gen = new_data["generation"]
        if gen["temperature"] <= 0:
            messagebox.showwarning("Invalid value", "generation.temperature must be greater than 0.")
            return
        if gen["top_k"] < 0:
            messagebox.showwarning("Invalid value", "generation.top_k must be >= 0.")
            return
        if not (0.0 <= gen["top_p"] <= 1.0):
            messagebox.showwarning("Invalid value", "generation.top_p must be between 0.0 and 1.0.")
            return
        if gen["repetition_penalty"] < 1.0:
            messagebox.showwarning("Invalid value", "generation.repetition_penalty must be >= 1.0.")
            return
        if gen["max_tokens"] < 1:
            messagebox.showwarning("Invalid value", "generation.max_tokens must be >= 1.")
            return
        tr = new_data["training"]
        if tr["lr_scheduler"] not in ("linear", "cosine_warmup", "constant"):
            messagebox.showwarning("Invalid value",
                                   "training.lr_scheduler must be linear, cosine_warmup or constant.")
            return
        if tr["min_lr"] < 0:
            messagebox.showwarning("Invalid value", "training.min_lr must be >= 0.")
            return
        if tr["grad_clip"] < 0:
            messagebox.showwarning("Invalid value", "training.grad_clip must be >= 0.")
            return
        if not (0.0 <= tr["val_split"] < 1.0):
            messagebox.showwarning("Invalid value", "training.val_split must be in [0.0, 1.0).")
            return
        if tr["min_delta"] < 0:
            messagebox.showwarning("Invalid value", "training.min_delta must be >= 0.")
            return
        if not isinstance(tr["early_stopping"], bool):
            messagebox.showwarning("Invalid value", "training.early_stopping must be true or false.")
            return
        if not (0.0 < tr["auto_factor"] < 1.0):
            messagebox.showwarning("Invalid value", "training.auto_factor must be between 0.0 and 1.0.")
            return
        if not isinstance(tr["auto_adjust"], bool) or not isinstance(tr["auto_clip_relax"], bool):
            messagebox.showwarning("Invalid value", "training.auto_adjust and auto_clip_relax must be true or false.")
            return

        try:
            _write_config_yaml(CONFIG_PATH, new_data)
            self.cfg = Config.from_dict(new_data)
        except Exception as exc:
            messagebox.showerror("Error", f"Failed to save config:\n{exc}")
            return

        # Sync quick-access entries with new defaults
        self.steps_var.delete(0, "end")
        self.steps_var.insert(0, str(new_data["training"]["steps"]))
        self.max_tokens_var.delete(0, "end")
        self.max_tokens_var.insert(0, str(new_data["generation"]["max_tokens"]))
        self.min_tokens_var.delete(0, "end")
        self.min_tokens_var.insert(0, str(new_data["generation"]["min_tokens"]))
        self.temp_var.delete(0, "end")
        self.temp_var.insert(0, str(new_data["generation"]["temperature"]))
        self.top_k_var.delete(0, "end")
        self.top_k_var.insert(0, str(new_data["generation"]["top_k"]))
        self.top_p_var.delete(0, "end")
        self.top_p_var.insert(0, str(new_data["generation"]["top_p"]))
        self.rep_penalty_var.delete(0, "end")
        self.rep_penalty_var.insert(0, str(new_data["generation"]["repetition_penalty"]))
        self.stop_tokens_var.delete(0, "end")
        self.stop_tokens_var.insert(0, ",".join(str(x) for x in new_data["generation"].get("stop_token_ids", [])))
        self.stop_strings_var.delete(0, "end")
        self.stop_strings_var.insert(0, ",".join(new_data["generation"].get("stop_strings", [])))

        self._log_unified("system", f"Config saved to {CONFIG_PATH}")

    # ---- tokenizer manager actions -------------------------------------------
