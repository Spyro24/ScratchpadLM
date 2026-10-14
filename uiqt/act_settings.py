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


class SettingsActionsMixin:
    def _browse_dir(self, line_edit):
        """Open directory browser for a line edit."""
        dir_path = QFileDialog.getExistingDirectory(self, "Select Directory", line_edit.text())
        if dir_path:
            line_edit.setText(dir_path)

    def _on_theme_change(self, theme: str):
        """Handle theme change from combobox."""
        if theme != self.current_theme:
            self.current_theme = theme
            QApplication.instance().setStyleSheet(qss_theme(theme))
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
            self.log_unified("system", f"Config saved from editor to {CONFIG_PATH}")
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
        for (section, key), (widget, _) in self._settings_entries.items():
            try:
                # Extract value based on widget type
                if isinstance(widget, QSpinBox):
                    value = widget.value()
                    # Special handling for optional fields where 0 means "auto/None"
                    if key in ("n_negatives_start", "workers", "restart_period", "cycle_length") and value == 0:
                        value = None
                elif isinstance(widget, QDoubleSpinBox):
                    value = widget.value()
                elif isinstance(widget, QComboBox):
                    value = widget.currentText()
                elif isinstance(widget, QCheckBox):
                    value = widget.isChecked()
                elif isinstance(widget, QLineEdit):
                    raw = widget.text().strip()
                    if key in ("stop_token_ids", "stop_strings"):
                        if key == "stop_token_ids":
                            value = [int(x.strip()) for x in raw.split(",") if x.strip()]
                        else:
                            value = [x.strip() for x in raw.split(",") if x.strip()]
                    else:
                        # Try to infer type from existing config
                        current_val = new_data[section][key]
                        if isinstance(current_val, int) and not isinstance(current_val, bool):
                            value = int(raw)
                        elif isinstance(current_val, float):
                            value = float(raw)
                        elif isinstance(current_val, bool):
                            value = raw.lower() in ("true", "1", "yes", "on")
                        else:
                            value = raw
                elif isinstance(widget, QWidget):
                    # Check if it's a wrapper with a line edit (for dir browser)
                    layout = widget.layout()
                    if layout:
                        line_edit = layout.itemAt(0).widget()
                        if isinstance(line_edit, QLineEdit):
                            value = line_edit.text().strip()
                        else:
                            value = ""
                    else:
                        continue
                else:
                    continue
                    
                # Special handling for None values
                if key in ("n_negatives_start", "workers") and hasattr(widget, 'value') and widget.value() == 0:
                    value = None
                    
            except ValueError:
                self.show_warning("Invalid value", f"{section}.{key}: failed to parse value.")
                return
                
            new_data[section][key] = value

        gen = new_data["generation"]
        checks = [
            (gen["max_tokens"] >= 0, "generation.max_tokens must be >= 0 (0 = unlimited)."),
            (gen["min_tokens"] >= 0, "generation.min_tokens must be >= 0."),
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
        self.log_unified("system", f"Config saved to {CONFIG_PATH}")

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
