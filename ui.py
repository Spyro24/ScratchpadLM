import os
import queue
import threading
import tkinter as tk
from tkinter import scrolledtext, messagebox

from tokenizer import BPETokenizer
from ai import TinyLM, train, generate, load_model, save_model, sync_vocab, train_parallel

MODEL_PATH = "tinyLM"
TOKENIZER_PATH = "bpe_merges.txt"
VOCAB_SIZE = 1000
CTX, EMB, HIDDEN = 50, 50, 50


class App:
    def __init__(self, root):
        self.root = root
        root.title("TinyLM - Prompt Interface")
        root.geometry("800x700")

        self.queue = queue.Queue()
        self.busy = False

        self.tok = BPETokenizer()
        if os.path.exists(TOKENIZER_PATH):
            self.tok.load(TOKENIZER_PATH)
        self.model = self._load_model()

        self._build_ui()
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

    # ---- UI ----------------------------------------------------------------

    def _build_ui(self):
        # Main container with padding
        main = tk.Frame(self.root, padx=10, pady=10)
        main.pack(fill="both", expand=True)
        main.rowconfigure(2, weight=1)  # Conversation history expands
        main.columnconfigure(0, weight=1)

        # ============================================================
        # SECTION 1: Training Data Input (top)
        # ============================================================
        train_frame = tk.LabelFrame(main, text="Training Data", padx=10, pady=10)
        train_frame.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        train_frame.columnconfigure(0, weight=1)

        # Training text area (multi-line for larger training data)
        self.train_text = scrolledtext.ScrolledText(train_frame, wrap="word", height=4)
        self.train_text.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        self.train_text.insert("1.0", "Enter training text here...")

        # Training controls row
        train_controls = tk.Frame(train_frame)
        train_controls.grid(row=1, column=0, sticky="ew")
        train_controls.columnconfigure(1, weight=1)

        tk.Label(train_controls, text="Steps:").grid(row=0, column=0, padx=(0, 5))
        self.steps_var = tk.Entry(train_controls, width=10)
        self.steps_var.insert(0, "100")
        self.steps_var.grid(row=0, column=1, sticky="w", padx=(0, 10))

        self.train_btn = tk.Button(train_controls, text="Train on Data", width=15, command=self.on_train)
        self.train_btn.grid(row=0, column=2, padx=5)

        # Vocab size display
        self.vocab_label = tk.Label(train_controls, text=f"Vocab: {len(self.tok.vocab)}")
        self.vocab_label.grid(row=0, column=3, padx=10)

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
        self.n_tokens_var.insert(0, "60")
        self.n_tokens_var.grid(row=0, column=1, padx=(0, 15))

        tk.Label(gen_settings, text="Temp:").grid(row=0, column=2, padx=(0, 5))
        self.temp_var = tk.Entry(gen_settings, width=8)
        self.temp_var.insert(0, "0.8")
        self.temp_var.grid(row=0, column=3, padx=(0, 15))

        self.clear_btn = tk.Button(gen_settings, text="Clear History", command=self.on_clear_history)
        self.clear_btn.grid(row=0, column=4, padx=10)

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
        self.tok.extend(prompt, vocab_size=VOCAB_SIZE, min_frequency=2)
        self.tok.save(TOKENIZER_PATH)
        sync_vocab(self.model, len(self.tok.vocab))

        ids = self.tok.encode(prompt)
        if len(ids) <= self.model.ctx:
            raise ValueError(
                f"The input has {len(ids)} tokens, but training needs more than {self.model.ctx}."
            )
        train_parallel(self.model, ids, steps=steps)
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


if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()