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
        root.title("TinyLM")
        root.geometry("700x560")

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
        top = tk.Frame(self.root, padx=10, pady=10)
        top.pack(fill="x")

        tk.Label(top, text="Input:").grid(row=0, column=0, sticky="w")
        self.input_var = tk.Entry(top, width=60)
        self.input_var.grid(row=0, column=1, sticky="we", padx=5)
        self.input_var.bind("<Return>", lambda e: self.on_generate())
        top.columnconfigure(1, weight=1)

        tk.Label(top, text="Training steps:").grid(row=1, column=0, sticky="w", pady=(8, 0))
        self.steps_var = tk.Entry(top, width=10)
        self.steps_var.insert(0, "100")
        self.steps_var.grid(row=1, column=1, sticky="w", pady=(8, 0))

        buttons = tk.Frame(self.root, padx=10)
        buttons.pack(fill="x")
        self.train_btn = tk.Button(buttons, text="Train", width=12, command=self.on_train)
        self.train_btn.pack(side="left")
        self.gen_btn = tk.Button(buttons, text="Generate", width=12, command=self.on_generate)
        self.gen_btn.pack(side="left", padx=8)

        self.status = tk.Label(buttons, text="Ready", anchor="w")
        self.status.pack(side="left", padx=10)

        tk.Label(self.root, text="History:", anchor="w", padx=10).pack(fill="x", pady=(10, 0))
        self.history = scrolledtext.ScrolledText(self.root, wrap="word", state="disabled")
        self.history.pack(fill="both", expand=True, padx=10, pady=(0, 10))

    def _append_history(self, text):
        self.history.configure(state="normal")
        self.history.insert("end", text + "\n\n")
        self.history.see("end")
        self.history.configure(state="disabled")

    def _set_busy(self, busy, message=""):
        self.busy = busy
        state = "disabled" if busy else "normal"
        self.train_btn.configure(state=state)
        self.gen_btn.configure(state=state)
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
            messagebox.showerror("Error", f"{label} failed:\n{payload}")
            return

        if label == "Training":
            prompt, steps = payload
            save_model(self.model, MODEL_PATH)
            self._append_history(f'[Trained on "{prompt}", {steps} steps, model saved]')
            self._set_busy(False, "Training done")
        elif label == "Generating":
            prompt, text = payload
            self._append_history(f"> {prompt}\n{text}")
            self._set_busy(False, "Ready")

    # ---- actions -----------------------------------------------------------

    def on_train(self):
        if self.busy:
            return
        prompt = self.input_var.get().strip()
        if not prompt:
            messagebox.showwarning("No input", "Type some text in the input field to train on.")
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

    def on_generate(self):
        if self.busy:
            return
        prompt = self.input_var.get()
        self._run_in_background("Generating", self._generate_job, prompt)

    def _generate_job(self, prompt):
        text = generate(self.model, self.tok, prompt, n_tokens=60)
        return prompt, text


if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()