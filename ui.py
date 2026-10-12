import os
import queue
import threading
import tkinter as tk
from tkinter import scrolledtext, messagebox, ttk, filedialog
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


class App:
    def __init__(self, root):
        self.root = root
        root.title("TinyLM - Prompt Interface")
        root.geometry("800x700")

        self.queue = queue.Queue()
        self.busy = False
        self.cfg = cfg  # Store config reference

        # Use shared tokenizer manager
        self.tok = TOKENIZER_MANAGER.tokenizer
        if os.path.exists(TOKENIZER_PATH):
            TOKENIZER_MANAGER.load(TOKENIZER_PATH)
        self.model = self._load_model()

        self._build_ui()
        self.update_model_inspector()
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
        main.rowconfigure(2, weight=1)  # Conversation history expands
        main.rowconfigure(3, weight=0)  # Model inspector (fixed height)
        main.columnconfigure(0, weight=1)

        # ============================================================
        # SECTION 1: Training Data Input (top)
        # ============================================================
        train_frame = tk.LabelFrame(main, text="Training Data", padx=10, pady=10)
        train_frame.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        train_frame.columnconfigure(0, weight=1)

        # Model selector row (above training text)
        model_row = tk.Frame(train_frame)
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

        # Training text area (multi-line for larger training data)
        self.train_text = scrolledtext.ScrolledText(train_frame, wrap="word", height=4)
        self.train_text.grid(row=1, column=0, sticky="ew", pady=(0, 8))
        self.train_text.insert("1.0", "Enter training text here...")

        # Training controls row
        train_controls = tk.Frame(train_frame)
        train_controls.grid(row=2, column=0, sticky="ew")
        train_controls.columnconfigure(1, weight=1)

        tk.Label(train_controls, text="Steps:").grid(row=0, column=0, padx=(0, 5))
        self.steps_var = tk.Entry(train_controls, width=10)
        self.steps_var.insert(0, str(self.cfg["training"]["steps"]))
        self.steps_var.grid(row=0, column=1, sticky="w", padx=(0, 10))

        self.train_btn = tk.Button(train_controls, text="Train on Data", width=15, command=self.on_train)
        self.train_btn.grid(row=0, column=2, padx=5)

        # ============================================================
        # SECTION 2: Conversation History (middle - expands)
        # ============================================================
        history_frame = tk.LabelFrame(main, text="Conversation", padx=10, pady=10)
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

        # ============================================================
        # SECTION 3: Prompt Input (bottom)
        # ============================================================
        prompt_frame = tk.LabelFrame(main, text="Prompt", padx=10, pady=10)
        prompt_frame.grid(row=2, column=0, sticky="ew")
        prompt_frame.columnconfigure(0, weight=1)

        # Prompt entry with Send button
        prompt_row = tk.Frame(prompt_frame)
        prompt_row.grid(row=0, column=0, sticky="ew")
        prompt_row.columnconfigure(0, weight=1)

        self.prompt_var = tk.Entry(prompt_row, font=("Consolas", 11))
        self.prompt_var.grid(row=0, column=0, sticky="ew", padx=(0, 8))
        self.prompt_var.bind("<Return>", lambda e: self.on_send_prompt())
        self.prompt_var.focus_set()

        self.send_btn = tk.Button(prompt_row, text="Send", width=10, command=self.on_send_prompt)
        self.send_btn.grid(row=0, column=1)

        # Generation settings row
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

        # ============================================================
        # SECTION 4: Model Inspector
        # ============================================================
        self.inspector_frame = tk.LabelFrame(main, text="Model Inspector", padx=10, pady=10)
        self.inspector_frame.grid(row=3, column=0, sticky="ew", pady=(10, 0))
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

        # Tensor details button
        self.tensor_detail_btn = tk.Button(self.inspector_frame, text="Show Tensor Details", command=self.on_show_tensor_details, state="disabled")
        self.tensor_detail_btn.grid(row=4, column=0, columnspan=4, pady=(10, 0))

        # Status bar at bottom
        self.status = tk.Label(self.root, text="Ready", anchor="w", bd=1, relief="sunken", padx=5)
        self.status.pack(fill="x", side="bottom", padx=10, pady=(0, 10))

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

    def _set_busy(self, busy, message=""):
        self.busy = busy
        state = "disabled" if busy else "normal"
        self.train_btn.configure(state=state)
        self.send_btn.configure(state=state)
        self.prompt_var.configure(state=state)
        self.clear_btn.configure(state=state)
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
                self._handle_result(kind, label, payload)
        except queue.Empty:
            pass
        self.root.after(100, self._poll_queue)

    def _handle_result(self, kind, label, payload):
        if kind == "error":
            self._set_busy(False, "Error")
            self._append_history("error", f"Error: {payload}")
            messagebox.showerror("Error", f"{label} failed:\n{payload}")
            return

        if label == "Training":
            prompt, steps = payload
            save_model(self.model, MODEL_PATH)
            self.vocab_label.configure(text=f"Vocab: {len(self.tok.vocab)}")
            self._append_history("system", f"[Trained on {len(prompt)} chars, {steps} steps, model saved. Vocab: {len(self.tok.vocab)}]")
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
        train_parallel(self.model, ids, steps=steps, lr=training_cfg["lr"], 
                       batch_size=training_cfg["batch_size"], n_negatives=training_cfg["n_negatives"],
                       workers=training_cfg["workers"])
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