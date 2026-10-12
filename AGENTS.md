# AGENTS.md

Guidance for AI coding agents working in this project.

## Project

TinyLM is a small pure-Python language model with no third-party dependencies.

- `tokenizer.py`: byte-level BPE tokenizer. `extend` adds tokens without changing existing IDs.
- `ai.py`: autograd (`Value`), `TinyLM`, training, generation, 4-bit save/load.
- `ui.py`: Tkinter app with input box, steps field, Train and Generate buttons, history.
- `tinyLM` and `bpe_merges.txt` are the saved model and tokenizer. Keep them together.

## Rules

- Standard library only. No NumPy or PyTorch.
- Never renumber existing token IDs.
- Never recreate the model when the vocabulary grows. Use `sync_vocab`.
- Only Tkinter calls on the main thread. Background work reports through `self.queue`.
- Multiprocessing code stays under `if __name__ == "__main__":`.
- Send only floats to worker processes, never `Value` objects.
- Update parameters in place (`p.data -= ...`). Do not replace `Value` objects.

## Do not

- Delete `bpe_merges.txt` or `tinyLM` unless the user asks. This discards the trained model.
- Add a folder-based corpus loader back into the UI.
