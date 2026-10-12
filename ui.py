import os
import queue
import threading
import time
import tkinter as tk
from tkinter import scrolledtext, messagebox, ttk, filedialog
from pathlib import Path
import json

from tokenizer import BPETokenizer
from ai import TinyLM, train, generate, load_model, save_model, sync_vocab, train_parallel, count_params
from registry import MODEL_REGISTRY, TASK_REGISTRY, TOKENIZER_MANAGER, get_tokenizer
from model_mgmt import ModelCatalog, ModelCreator, ModelLoader, MODEL_CATALOG
from config import Config

MODEL_PATH = "tinyLM"
TOKENIZER_PATH = "bpe_merges.txt"
CONFIG_PATH = "config.yaml"

# Load configuration
try:
    cfg = Config.from_file(CONFIG_PATH)
except Exception:
    cfg = Config.from_dict({})  # Use defaults

VOCAB_SIZE = cfg["model"]["vocab_size"]
CTX = cfg["model"]["ctx"]
EMB = cfg["model"]["emb"]
HIDDEN = cfg["model"]["hidden"]

# Available models from registry (updated dynamically)
def get_available_models():
    """Get list of available model names from registry, catalog, and local files."""
    models = list(MODEL_REGISTRY.list())
    # Add models from catalog that have valid files
    try:
        from model_mgmt import MODEL_CATALOG
        from pathlib import Path
        for spec in MODEL_CATALOG.list():
            if spec.name not in models and Path(spec.path).exists():
                models.append(spec.name)
    except Exception:
        pass
    # Add built-in if not present
    if "TinyLM" not in models:
        models.insert(0, "TinyLM")
    return models


def _write_config_yaml(path, data):
    """Write a 2-level config dict as YAML (matches config.py's parser)."""
    lines = ["# TinyLM Configuration"]
    for section, values in data.items():
        lines.append(f"{section}:")
        for key, value in values.items():
            if value is None:
                rendered = "null"
            elif isinstance(value, bool):
                rendered = "true" if value else "false"
            else:
                rendered = str(value)
            lines.append(f"  {key}: {rendered}")
        lines.append("")
    Path(path).write_text("\n".join(lines), encoding="utf-8")


class App:
    def __init__(self, root):
        self.root = root
        root.title("ScratchpadLM (TinyLM)")
        root.geometry("900x720")

        self.queue = queue.Queue()
        self.busy = False
        self.cfg = cfg  # Store config reference

        # Use shared tokenizer manager
        self.tok = TOKENIZER_MANAGER.tokenizer
        if os.path.exists(TOKENIZER_PATH):
            TOKENIZER_MANAGER.load(TOKENIZER_PATH)
        self.model = self._load_model()

        self._catalog_names = []  # names shown in Admin listbox (parallel order)

        self._build_ui()
        self.update_model_inspector()
        self.refresh_catalog_list()
        self.root.after(100, self._poll_queue)

    # ---- setup -------------------------------------------------------------

    def _load_model(self):
        """Load the saved model if it matches the tokenizer, else create a new one."""
        try:
            model = load_model(MODEL_PATH)
            if len(model.C) == len(self.tok.vocab):
                return model
        except Exception:
            pass
        return TinyLM(vocab_size=len(self.tok.vocab), ctx=CTX, emb=EMB, hidden=HIDDEN)

    def refresh_models(self):
        """Refresh the model dropdown with available models."""
        models = get_available_models()
        self.model_combo["values"] = models
        if self.model_var.get() not in models:
            self.model_var.set(models[0] if models else "TinyLM")
        self._append_history("system", f"[Refreshed model list: {', '.join(models)}]")

    def on_model_change(self, event=None):
        """Handle model selection change."""
        model_name = self.model_var.get()
        self._switch_model(model_name)

    def _switch_model(self, model_name):
        """Switch to the selected model."""
        if self.busy:
            return
        self._append_history("system", f"[Switching to model: {model_name}...]")
        
        # Run model loading in background
        self._run_in_background(f"Loading {model_name}", self._load_model_job, model_name)

    def _load_model_job(self, model_name):
        """Background job to load a model."""
        if model_name == "TinyLM":
            # Create a fresh TinyLM with current vocab size
            return TinyLM(vocab_size=len(self.tok.vocab), ctx=CTX, emb=EMB, hidden=HIDDEN)
        try:
            factory = MODEL_REGISTRY.get(model_name)
            return factory()
        except KeyError:
            # Try loading from catalog
            try:
                from model_mgmt import ModelLoader, MODEL_CATALOG
                loader = ModelLoader(MODEL_CATALOG, tokenizer_manager=TOKENIZER_MANAGER)
                return loader.load_model(model_name)
            except Exception as e:
                raise ValueError(f"Model '{model_name}' not found in registry or catalog: {e}")

    def _get_model_factory(self, model_name):
        """Get model factory from registry or return TinyLM."""
        if model_name == "TinyLM":
            return lambda: TinyLM(vocab_size=len(self.tok.vocab), ctx=CTX, emb=EMB, hidden=HIDDEN)
        try:
            return MODEL_REGISTRY.get(model_name)
        except KeyError:
            return None

    # ---- UI ----------------------------------------------------------------

    def _build_ui(self):
        # Main container with padding
        main = tk.Frame(self.root, padx=10, pady=10)
        main.pack(fill="both", expand=True)
        main.rowconfigure(0, weight=1)
        main.columnconfigure(0, weight=1)

        # Tabbed layout (roadmap 0a: Chat, Training, Model Info, Admin, Settings)
        self.notebook = ttk.Notebook(main)
        self.notebook.grid(row=0, column=0, sticky="nsew")

        chat_tab = tk.Frame(self.notebook, padx=10, pady=10)
        training_tab = tk.Frame(self.notebook, padx=10, pady=10)
        info_tab = tk.Frame(self.notebook, padx=10, pady=10)
        admin_tab = tk.Frame(self.notebook, padx=10, pady=10)
        settings_tab = tk.Frame(self.notebook, padx=10, pady=10)

        self.notebook.add(chat_tab, text="Chat")
        self.notebook.add(training_tab, text="Training")
        self.notebook.add(info_tab, text="Model Info")
        self.notebook.add(admin_tab, text="Admin")
        self.notebook.add(settings_tab, text="Settings")

        self._build_chat_tab(chat_tab)
        self._build_training_tab(training_tab)
        self._build_model_info_tab(info_tab)
        self._build_admin_tab(admin_tab)
        self._build_settings_tab(settings_tab)

        # Status bar at bottom
        self.status = tk.Label(self.root, text="Ready", anchor="w", bd=1, relief="sunken", padx=5)
        self.status.pack(fill="x", side="bottom", padx=10, pady=(0, 10))

    def _build_chat_tab(self, tab):
        """Chat tab: model selector, conversation history, prompt input."""
        tab.rowconfigure(1, weight=1)  # History expands
        tab.columnconfigure(0, weight=1)

        # Model selector row
        model_row = tk.Frame(tab)
        model_row.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        model_row.columnconfigure(1, weight=1)

        tk.Label(model_row, text="Model:").grid(row=0, column=0, padx=(0, 5))
        self.model_var = tk.StringVar(value="TinyLM")
        self.model_combo = ttk.Combobox(model_row, textvariable=self.model_var, state="readonly", width=20)
        self.model_combo["values"] = get_available_models()
        self.model_combo.grid(row=0, column=1, sticky="w", padx=(0, 10))
        self.model_combo.bind("<<ComboboxSelected>>", self.on_model_change)

        # Refresh models button
        tk.Button(model_row, text="Refresh Models", command=self.refresh_models, width=15).grid(row=0, column=2, padx=5)

        # Vocab size display
        self.vocab_label = tk.Label(model_row, text=f"Vocab: {len(self.tok.vocab)}")
        self.vocab_label.grid(row=0, column=3, padx=10)

        # Conversation history
        history_frame = tk.LabelFrame(tab, text="Conversation", padx=10, pady=10)
        history_frame.grid(row=1, column=0, sticky="nsew", pady=(0, 10))
        history_frame.rowconfigure(0, weight=1)
        history_frame.columnconfigure(0, weight=1)

        self.history = scrolledtext.ScrolledText(
            history_frame, wrap="word", state="disabled", font=("Consolas", 10)
        )
        self.history.grid(row=0, column=0, sticky="nsew")

        # Configure tags for styling
        self.history.tag_configure("user", foreground="#2E86C1", font=("Consolas", 10, "bold"))
        self.history.tag_configure("assistant", foreground="#27AE60", font=("Consolas", 10, "bold"))
        self.history.tag_configure("system", foreground="#7F8C8D", font=("Consolas", 9, "italic"))
        self.history.tag_configure("error", foreground="#E74C3C", font=("Consolas", 10, "bold"))

        # Prompt input
        prompt_frame = tk.LabelFrame(tab, text="Prompt", padx=10, pady=10)
        prompt_frame.grid(row=2, column=0, sticky="ew")
        prompt_frame.columnconfigure(0, weight=1)

        prompt_row = tk.Frame(prompt_frame)
        prompt_row.grid(row=0, column=0, sticky="ew")
        prompt_row.columnconfigure(0, weight=1)

        self.prompt_var = tk.Entry(prompt_row, font=("Consolas", 11))
        self.prompt_var.grid(row=0, column=0, sticky="ew", padx=(0, 8))
        self.prompt_var.bind("<Return>", lambda e: self.on_send_prompt())
        self.prompt_var.focus_set()

        self.send_btn = tk.Button(prompt_row, text="Send", width=10, command=self.on_send_prompt)
        self.send_btn.grid(row=0, column=1)

        # Quick generation settings row
        gen_settings = tk.Frame(prompt_frame)
        gen_settings.grid(row=1, column=0, sticky="ew", pady=(8, 0))

        tk.Label(gen_settings, text="Tokens:").grid(row=0, column=0, padx=(0, 5))
        self.n_tokens_var = tk.Entry(gen_settings, width=8)
        self.n_tokens_var.insert(0, str(self.cfg["generation"]["n_tokens"]))
        self.n_tokens_var.grid(row=0, column=1, padx=(0, 15))

        tk.Label(gen_settings, text="Temp:").grid(row=0, column=2, padx=(0, 5))
        self.temp_var = tk.Entry(gen_settings, width=8)
        self.temp_var.insert(0, str(self.cfg["generation"]["temperature"]))
        self.temp_var.grid(row=0, column=3, padx=(0, 15))

        self.clear_btn = tk.Button(gen_settings, text="Clear History", command=self.on_clear_history)
        self.clear_btn.grid(row=0, column=4, padx=10)

    def _build_training_tab(self, tab):
        """Training tab: data input, controls, training log monitor."""
        tab.rowconfigure(1, weight=1)  # Training text expands
        tab.rowconfigure(4, weight=1)  # Log monitor expands
        tab.columnconfigure(0, weight=1)

        # Target model indicator (mirrors Chat tab selection)
        target_row = tk.Frame(tab)
        target_row.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        tk.Label(target_row, text="Training target:", font=("Consolas", 9, "bold")).pack(side="left")
        tk.Label(target_row, textvariable=self.model_var, font=("Consolas", 9)).pack(side="left", padx=(5, 0))

        # Training text area
        train_frame = tk.LabelFrame(tab, text="Training Data", padx=10, pady=10)
        train_frame.grid(row=1, column=0, sticky="nsew", pady=(0, 10))
        train_frame.rowconfigure(0, weight=1)
        train_frame.columnconfigure(0, weight=1)

        self.train_text = scrolledtext.ScrolledText(train_frame, wrap="word", height=6)
        self.train_text.grid(row=0, column=0, sticky="nsew")
        self.train_text.insert("1.0", "Enter training text here...")

        # Training controls row
        train_controls = tk.Frame(tab)
        train_controls.grid(row=2, column=0, sticky="ew", pady=(0, 10))

        tk.Label(train_controls, text="Steps:").pack(side="left", padx=(0, 5))
        self.steps_var = tk.Entry(train_controls, width=10)
        self.steps_var.insert(0, str(self.cfg["training"]["steps"]))
        self.steps_var.pack(side="left", padx=(0, 10))

        self.train_btn = tk.Button(train_controls, text="Train on Data", width=15, command=self.on_train)
        self.train_btn.pack(side="left", padx=5)

        # Progress bar row
        progress_row = tk.Frame(tab)
        progress_row.grid(row=3, column=0, sticky="ew", pady=(8, 0))
        progress_row.columnconfigure(0, weight=1)

        self.train_progress = ttk.Progressbar(progress_row, mode="determinate", maximum=100)
        self.train_progress.grid(row=0, column=0, sticky="ew", padx=(0, 8))
        self.progress_label = tk.Label(progress_row, text="", font=("Consolas", 9), width=22, anchor="w")
        self.progress_label.grid(row=0, column=1)

        # Training monitor (log)
        monitor_frame = tk.LabelFrame(tab, text="Training Monitor", padx=10, pady=10)
        monitor_frame.grid(row=4, column=0, sticky="nsew")
        monitor_frame.rowconfigure(0, weight=1)
        monitor_frame.columnconfigure(0, weight=1)

        self.train_log = scrolledtext.ScrolledText(
            monitor_frame, wrap="word", state="disabled", height=8, font=("Consolas", 9)
        )
        self.train_log.grid(row=0, column=0, sticky="nsew")
        self.train_log.tag_configure("ts", foreground="#7F8C8D")
        self.train_log.tag_configure("err", foreground="#E74C3C")

    def _build_model_info_tab(self, tab):
        """Model Info tab: inspector + tensor details."""
        self.inspector_frame = tk.LabelFrame(tab, text="Model Inspector", padx=10, pady=10)
        self.inspector_frame.pack(fill="x")
        self.inspector_frame.columnconfigure(1, weight=1)
        self.inspector_frame.columnconfigure(3, weight=1)

        # Model name
        tk.Label(self.inspector_frame, text="Model:", font=("Consolas", 9, "bold")).grid(row=0, column=0, sticky="w", padx=(0, 5))
        self.inspector_model_name = tk.Label(self.inspector_frame, text="None", font=("Consolas", 9))
        self.inspector_model_name.grid(row=0, column=1, sticky="w", padx=(0, 20))

        # Architecture
        tk.Label(self.inspector_frame, text="Architecture:", font=("Consolas", 9, "bold")).grid(row=0, column=2, sticky="w", padx=(0, 5))
        self.inspector_arch = tk.Label(self.inspector_frame, text="Unknown", font=("Consolas", 9))
        self.inspector_arch.grid(row=0, column=3, sticky="w")

        # Parameter count
        tk.Label(self.inspector_frame, text="Parameters:", font=("Consolas", 9, "bold")).grid(row=1, column=0, sticky="w", padx=(0, 5), pady=(5, 0))
        self.inspector_params = tk.Label(self.inspector_frame, text="0", font=("Consolas", 9))
        self.inspector_params.grid(row=1, column=1, sticky="w", padx=(0, 20), pady=(5, 0))

        # Memory estimate (4-bit quantized)
        tk.Label(self.inspector_frame, text="Memory (4-bit):", font=("Consolas", 9, "bold")).grid(row=1, column=2, sticky="w", padx=(0, 5), pady=(5, 0))
        self.inspector_memory = tk.Label(self.inspector_frame, text="0 KB", font=("Consolas", 9))
        self.inspector_memory.grid(row=1, column=3, sticky="w", pady=(5, 0))

        # Vocab size
        tk.Label(self.inspector_frame, text="Vocab Size:", font=("Consolas", 9, "bold")).grid(row=2, column=0, sticky="w", padx=(0, 5), pady=(5, 0))
        self.inspector_vocab = tk.Label(self.inspector_frame, text="0", font=("Consolas", 9))
        self.inspector_vocab.grid(row=2, column=1, sticky="w", padx=(0, 20), pady=(5, 0))

        # Context size
        tk.Label(self.inspector_frame, text="Context:", font=("Consolas", 9, "bold")).grid(row=2, column=2, sticky="w", padx=(0, 5), pady=(5, 0))
        self.inspector_ctx = tk.Label(self.inspector_frame, text="0", font=("Consolas", 9))
        self.inspector_ctx.grid(row=2, column=3, sticky="w", pady=(5, 0))

        # Embedding dim
        tk.Label(self.inspector_frame, text="Embedding:", font=("Consolas", 9, "bold")).grid(row=3, column=0, sticky="w", padx=(0, 5), pady=(5, 0))
        self.inspector_emb = tk.Label(self.inspector_frame, text="0", font=("Consolas", 9))
        self.inspector_emb.grid(row=3, column=1, sticky="w", padx=(0, 20), pady=(5, 0))

        # Hidden size
        tk.Label(self.inspector_frame, text="Hidden:", font=("Consolas", 9, "bold")).grid(row=3, column=2, sticky="w", padx=(0, 5), pady=(5, 0))
        self.inspector_hidden = tk.Label(self.inspector_frame, text="0", font=("Consolas", 9))
        self.inspector_hidden.grid(row=3, column=3, sticky="w", pady=(5, 0))

        # Buttons: tensor details popup + manual refresh
        btn_row = tk.Frame(tab)
        btn_row.pack(pady=(10, 0))
        self.tensor_detail_btn = tk.Button(btn_row, text="Show Tensor Details", command=self.on_show_tensor_details, state="disabled")
        self.tensor_detail_btn.pack(side="left", padx=5)
        tk.Button(btn_row, text="Refresh", command=self.update_model_inspector).pack(side="left", padx=5)

    def _build_admin_tab(self, tab):
        """Admin tab: catalog model list, create/delete, catalog mgmt."""
        tab.rowconfigure(1, weight=1)
        tab.columnconfigure(0, weight=1)

        # Catalog list
        list_frame = tk.LabelFrame(tab, text="Model Catalog", padx=10, pady=10)
        list_frame.grid(row=1, column=0, sticky="nsew")
        list_frame.rowconfigure(0, weight=1)
        list_frame.columnconfigure(0, weight=1)

        self.admin_list = tk.Listbox(list_frame, font=("Consolas", 9))
        self.admin_list.grid(row=0, column=0, sticky="nsew")
        scrollbar = tk.Scrollbar(list_frame, orient="vertical", command=self.admin_list.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.admin_list.configure(yscrollcommand=scrollbar.set)

        # Refresh catalog button
        tk.Button(tab, text="Refresh Catalog", command=self.refresh_catalog_list, width=15).grid(row=2, column=0, pady=(8, 0))

        # Create model row
        create_frame = tk.LabelFrame(tab, text="Create Model", padx=10, pady=10)
        create_frame.grid(row=3, column=0, sticky="ew", pady=(10, 0))
        create_frame.columnconfigure(1, weight=1)

        tk.Label(create_frame, text="Name:").grid(row=0, column=0, padx=(0, 5))
        self.create_name_var = tk.Entry(create_frame, width=20)
        self.create_name_var.grid(row=0, column=1, sticky="w", padx=(0, 10))

        tk.Label(create_frame, text="Domains:").grid(row=0, column=2, padx=(0, 5))
        self.create_domains_var = tk.Entry(create_frame, width=30)
        self.create_domains_var.grid(row=0, column=3, sticky="w", padx=(0, 10))

        self.create_btn = tk.Button(create_frame, text="Create", command=self.on_create_model)
        self.create_btn.grid(row=0, column=4)

        # Delete model row
        delete_frame = tk.LabelFrame(tab, text="Delete Model", padx=10, pady=10)
        delete_frame.grid(row=4, column=0, sticky="ew", pady=(10, 0))

        self.delete_btn = tk.Button(delete_frame, text="Delete Selected", command=self.on_delete_model)
        self.delete_btn.pack(side="left")

    def _build_settings_tab(self, tab):
        """Settings tab: edit generation/training/tokenizer/model config, save to config.yaml."""
        tab.columnconfigure(1, weight=1)

        # (section, key, label, type) for each editable setting
        settings = [
            ("generation", "n_tokens", "Generation tokens", int),
            ("generation", "temperature", "Temperature", float),
            ("training", "steps", "Training steps", int),
            ("training", "lr", "Learning rate", float),
            ("training", "batch_size", "Batch size", int),
            ("training", "n_negatives", "Negatives", int),
            ("training", "workers", "Workers", int),
            ("tokenizer", "vocab_size", "Tokenizer vocab size", int),
            ("tokenizer", "min_frequency", "Min frequency", int),
            ("model", "vocab_size", "Model vocab size", int),
            ("model", "ctx", "Context length", int),
            ("model", "emb", "Embedding dim", int),
            ("model", "hidden", "Hidden dim", int),
        ]

        self._settings_entries = {}  # (section, key) -> (Entry, type)

        row = 0
        current_section = None
        for section, key, label, cast in settings:
            if section != current_section:
                current_section = section
                tk.Label(tab, text=section.upper(), font=("Consolas", 10, "bold")).grid(
                    row=row, column=0, columnspan=2, sticky="w", pady=(10 if row else 0, 2)
                )
                row += 1
            tk.Label(tab, text=label + ":").grid(row=row, column=0, sticky="w", padx=(10, 5))
            entry = tk.Entry(tab, width=12)
            entry.insert(0, str(self.cfg[section][key]))
            entry.grid(row=row, column=1, sticky="w", pady=1)
            self._settings_entries[(section, key)] = (entry, cast)
            row += 1

        tk.Button(tab, text="Save Config", command=self.on_save_config, width=15).grid(
            row=row + 1, column=0, columnspan=2, pady=(15, 0)
        )

    def _append_history(self, role, text):
        """Append a message to conversation history with role styling."""
        self.history.configure(state="normal")
        
        if role == "user":
            self.history.insert("end", "You: ", "user")
            self.history.insert("end", text + "\n\n")
        elif role == "assistant":
            self.history.insert("end", "AI: ", "assistant")
            self.history.insert("end", text + "\n\n")
        elif role == "system":
            self.history.insert("end", text + "\n\n", "system")
        elif role == "error":
            self.history.insert("end", text + "\n\n", "error")
        
        self.history.see("end")
        self.history.configure(state="disabled")

    def _log_train(self, message, is_error=False):
        """Append a timestamped line to the training monitor."""
        ts = time.strftime("%H:%M:%S")
        self.train_log.configure(state="normal")
        self.train_log.insert("end", f"[{ts}] ", "ts")
        self.train_log.insert("end", message + "\n", "err" if is_error else "")
        self.train_log.see("end")
        self.train_log.configure(state="disabled")

    def _set_busy(self, busy, message=""):
        self.busy = busy
        state = "disabled" if busy else "normal"
        self.train_btn.configure(state=state)
        self.send_btn.configure(state=state)
        self.prompt_var.configure(state=state)
        self.clear_btn.configure(state=state)
        self.create_btn.configure(state=state)
        self.delete_btn.configure(state=state)
        self.status.configure(text=message or ("Working..." if busy else "Ready"))

    # ---- background work ---------------------------------------------------

    def _run_in_background(self, label, func, *args):
        self._set_busy(True, f"{label}...")

        def worker():
            try:
                result = func(*args)
                self.queue.put(("ok", label, result))
            except Exception as exc:
                self.queue.put(("error", label, exc))

        threading.Thread(target=worker, daemon=True).start()

    def _poll_queue(self):
        try:
            while True:
                kind, label, payload = self.queue.get_nowait()
                if kind == "progress":
                    self._handle_progress(label, payload)
                else:
                    self._handle_result(kind, label, payload)
        except queue.Empty:
            pass
        self.root.after(100, self._poll_queue)

    def _handle_progress(self, label, payload):
        """Update the training progress bar (runs on the main thread)."""
        if label != "Training":
            return
        done, total, loss = payload
        self.train_progress.configure(maximum=max(total, 1), value=done)
        self.progress_label.configure(text=f"Step {done}/{total}  loss {loss:.3f}")

    def _handle_result(self, kind, label, payload):
        if kind == "error":
            self._set_busy(False, "Error")
            self._append_history("error", f"Error: {payload}")
            if label == "Training":
                self._log_train(f"{label} failed: {payload}", is_error=True)
                self.progress_label.configure(text="Failed")
            messagebox.showerror("Error", f"{label} failed:\n{payload}")
            return

        if label == "Training":
            prompt, steps = payload
            save_model(self.model, MODEL_PATH)
            self.vocab_label.configure(text=f"Vocab: {len(self.tok.vocab)}")
            self._append_history("system", f"[Trained on {len(prompt)} chars, {steps} steps, model saved. Vocab: {len(self.tok.vocab)}]")
            self._log_train(f"Training done: {steps} steps on {len(prompt)} chars, model saved, vocab={len(self.tok.vocab)}")
            self.train_progress.configure(maximum=max(steps, 1), value=steps)
            self.progress_label.configure(text=f"Done ({steps} steps)")
            self._set_busy(False, "Training done")
            self.update_model_inspector()
        elif label.startswith("Loading "):
            self.model = payload
            model_name = label.replace("Loading ", "")
            self._append_history("system", f"[Loaded model: {model_name} ({len(self.model.C)} vocab, {self.model.ctx} ctx)]")
            self._set_busy(False, "Ready")
            self.update_model_inspector()
        elif label == "Generating":
            prompt, text = payload
            self._append_history("assistant", text)
            self._set_busy(False, "Ready")
        elif label == "Creating model":
            spec = payload
            self._append_history("system", f"[Created model '{spec.name}' ({spec.param_count:,} params, domains: {', '.join(spec.domain_keywords)})]")
            self.refresh_catalog_list()
            self.refresh_models()
            self._set_busy(False, "Model created")
        elif label == "Deleting model":
            name, removed = payload
            if removed:
                self._append_history("system", f"[Deleted model '{name}' from catalog]")
            else:
                self._append_history("system", f"[Model '{name}' was not in catalog]")
            self.refresh_catalog_list()
            self.refresh_models()
            self._set_busy(False, "Model deleted")

    # ---- actions -----------------------------------------------------------

    def on_train(self):
        if self.busy:
            return
        prompt = self.train_text.get("1.0", "end-1c").strip()
        if not prompt or prompt == "Enter training text here...":
            messagebox.showwarning("No input", "Enter some training text in the Training Data area.")
            return
        try:
            steps = int(self.steps_var.get())
            if steps < 1:
                raise ValueError
        except ValueError:
            messagebox.showwarning("Invalid input", "Training steps must be a positive whole number.")
            return
        self._log_train(f"Training started: {len(prompt)} chars, {steps} steps, model={self.model_var.get()}")
        self.train_progress.configure(maximum=steps, value=0)
        self.progress_label.configure(text=f"Step 0/{steps}")
        self._run_in_background("Training", self._train_job, prompt, steps)

    def _train_job(self, prompt, steps):
        # Use config for tokenizer and training settings
        tokenizer_cfg = self.cfg["tokenizer"]
        training_cfg = self.cfg["training"]
        
        self.tok.extend(prompt, vocab_size=tokenizer_cfg["vocab_size"], min_frequency=tokenizer_cfg["min_frequency"])
        self.tok.save(TOKENIZER_PATH)
        sync_vocab(self.model, len(self.tok.vocab))

        ids = self.tok.encode(prompt)
        if len(ids) <= self.model.ctx:
            raise ValueError(
                f"The input has {len(ids)} tokens, but training needs more than {self.model.ctx}."
            )

        # Progress callback: thread-safe, only puts messages on the queue
        def on_progress(done, total, loss):
            self.queue.put(("progress", "Training", (done, total, loss)))

        train_parallel(self.model, ids, steps=steps, lr=training_cfg["lr"], 
                       batch_size=training_cfg["batch_size"], n_negatives=training_cfg["n_negatives"],
                       workers=training_cfg["workers"], progress_cb=on_progress)
        return prompt, steps

    def on_send_prompt(self):
        if self.busy:
            return
        prompt = self.prompt_var.get().strip()
        if not prompt:
            return
        # Clear prompt entry
        self.prompt_var.delete(0, "end")
        # Show user message immediately
        self._append_history("user", prompt)
        # Get generation settings
        try:
            n_tokens = int(self.n_tokens_var.get())
            temperature = float(self.temp_var.get())
        except ValueError:
            n_tokens, temperature = 60, 0.8
        self._run_in_background("Generating", self._generate_job, prompt, n_tokens, temperature)

    def _generate_job(self, prompt, n_tokens, temperature):
        text = generate(self.model, self.tok, prompt, n_tokens=n_tokens, temperature=temperature)
        return prompt, text

    def on_clear_history(self):
        self.history.configure(state="normal")
        self.history.delete("1.0", "end")
        self.history.configure(state="disabled")

    # ---- admin actions -----------------------------------------------------

    def refresh_catalog_list(self):
        """Refresh the Admin tab listbox from the model catalog."""
        self.admin_list.delete(0, "end")
        self._catalog_names = []
        for spec in MODEL_CATALOG.list():
            domains = ", ".join(spec.domain_keywords) or "-"
            self.admin_list.insert("end", f"{spec.name}  |  {spec.param_count:,} params  |  {domains}")
            self._catalog_names.append(spec.name)

    def on_create_model(self):
        if self.busy:
            return
        name = self.create_name_var.get().strip()
        if not name:
            messagebox.showwarning("No name", "Enter a model name (e.g. tiny-code).")
            return
        if MODEL_CATALOG.get(name) is not None:
            messagebox.showwarning("Exists", f"Model '{name}' already exists in the catalog.")
            return
        domains_raw = self.create_domains_var.get().strip()
        domains = [d.strip() for d in domains_raw.split(",") if d.strip()] or [name]
        # Use current config.yaml model section as base
        base_config = {
            "ctx": self.cfg["model"]["ctx"],
            "emb": self.cfg["model"]["emb"],
            "hidden": self.cfg["model"]["hidden"],
        }
        self._append_history("system", f"[Creating model '{name}' (domains: {', '.join(domains)})...]")
        self._run_in_background("Creating model", self._create_model_job, name, domains, base_config)

    def _create_model_job(self, name, domains, base_config):
        creator = ModelCreator(MODEL_CATALOG, "models", tokenizer_manager=TOKENIZER_MANAGER)
        return creator.create_model(name, domains, base_config=base_config)

    def on_delete_model(self):
        if self.busy:
            return
        selection = self.admin_list.curselection()
        if not selection:
            messagebox.showwarning("No selection", "Select a model in the catalog list first.")
            return
        name = self._catalog_names[selection[0]]
        if not messagebox.askyesno("Confirm delete", f"Delete model '{name}' from the catalog?\n(The .tlm file is removed too.)"):
            return
        self._run_in_background("Deleting model", self._delete_model_job, name)

    def _delete_model_job(self, name):
        spec = MODEL_CATALOG.get(name)
        removed = MODEL_CATALOG.remove(name)
        if spec is not None and os.path.exists(spec.path):
            try:
                os.remove(spec.path)
            except OSError:
                pass
        return name, removed

    # ---- settings actions --------------------------------------------------

    def on_save_config(self):
        """Validate Settings tab entries and save to config.yaml."""
        new_data = self.cfg.to_dict()
        for (section, key), (entry, cast) in self._settings_entries.items():
            raw = entry.get().strip()
            try:
                value = cast(raw)
            except ValueError:
                messagebox.showwarning(
                    "Invalid value", f"{section}.{key}: '{raw}' is not a valid {cast.__name__}."
                )
                return
            if isinstance(value, int) and value < 1:
                messagebox.showwarning("Invalid value", f"{section}.{key} must be a positive whole number.")
                return
            new_data[section][key] = value
        if new_data["generation"]["temperature"] <= 0:
            messagebox.showwarning("Invalid value", "generation.temperature must be greater than 0.")
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
        self.n_tokens_var.delete(0, "end")
        self.n_tokens_var.insert(0, str(new_data["generation"]["n_tokens"]))
        self.temp_var.delete(0, "end")
        self.temp_var.insert(0, str(new_data["generation"]["temperature"]))

        self._append_history("system", f"[Config saved to {CONFIG_PATH}]")

    def update_model_inspector(self):
        """Update the model inspector with current model details."""
        if self.model is None:
            self.inspector_model_name.configure(text="None")
            self.inspector_arch.configure(text="Unknown")
            self.inspector_params.configure(text="0")
            self.inspector_memory.configure(text="0 KB")
            self.inspector_vocab.configure(text="0")
            self.inspector_ctx.configure(text="0")
            self.inspector_emb.configure(text="0")
            self.inspector_hidden.configure(text="0")
            self.tensor_detail_btn.configure(state="disabled")
            return

        # Get model config
        config = self.model.get_config()
        arch = config.get("arch", "TinyLM")
        vocab_size = config.get("vocab_size", len(self.model.C))
        ctx = config.get("ctx", getattr(self.model, "ctx", 0))
        emb = config.get("emb", len(self.model.C[0]) if self.model.C else 0)
        hidden = config.get("hidden", len(self.model.b1) if self.model.b1 else 0)

        # Count parameters
        try:
            param_count = count_params(self.model)
        except Exception:
            param_count = 0

        # Estimate 4-bit quantized memory (4 bits per param = 0.5 bytes)
        memory_bytes = param_count * 0.5
        if memory_bytes >= 1024 * 1024:
            memory_str = f"{memory_bytes / (1024 * 1024):.2f} MB"
        else:
            memory_str = f"{memory_bytes / 1024:.1f} KB"

        # Update UI
        model_name = self.model_var.get()
        self.inspector_model_name.configure(text=model_name)
        self.inspector_arch.configure(text=arch)
        self.inspector_params.configure(text=f"{param_count:,}")
        self.inspector_memory.configure(text=memory_str)
        self.inspector_vocab.configure(text=str(vocab_size))
        self.inspector_ctx.configure(text=str(ctx))
        self.inspector_emb.configure(text=str(emb))
        self.inspector_hidden.configure(text=str(hidden))
        self.tensor_detail_btn.configure(state="normal")

    def on_show_tensor_details(self):
        """Show detailed tensor information in a popup window."""
        if self.model is None:
            return

        config = self.model.get_config()
        tensors = self.model.get_tensors()

        # Create detail window
        detail_win = tk.Toplevel(self.root)
        detail_win.title(f"Tensor Details - {self.model_var.get()}")
        detail_win.geometry("600x500")
        detail_win.transient(self.root)

        # Scrolled text for details
        text_area = scrolledtext.ScrolledText(detail_win, wrap="word", font=("Consolas", 9))
        text_area.pack(fill="both", expand=True, padx=10, pady=10)

        # Build detail text
        lines = []
        lines.append(f"Model: {self.model_var.get()}")
        lines.append(f"Architecture: {config.get('arch', 'Unknown')}")
        lines.append(f"Vocab Size: {config.get('vocab_size', 'N/A')}")
        lines.append(f"Context: {config.get('ctx', 'N/A')}")
        lines.append(f"Embedding Dim: {config.get('emb', 'N/A')}")
        lines.append(f"Hidden Dim: {config.get('hidden', 'N/A')}")
        lines.append("")

        total_params = 0
        for name, rows in tensors.items():
            n_rows = len(rows)
            n_cols = len(rows[0]) if rows else 0
            tensor_params = n_rows * n_cols
            total_params += tensor_params
            lines.append(f"  {name}: {n_rows} x {n_cols} = {tensor_params:,} params")

        lines.append("")
        lines.append(f"Total Parameters: {total_params:,}")

        text_area.insert("1.0", "\n".join(lines))
        text_area.configure(state="disabled")

        # Close button
        tk.Button(detail_win, text="Close", command=detail_win.destroy).pack(pady=(0, 10))


if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()
