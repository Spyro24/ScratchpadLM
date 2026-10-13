"""Shared Tkinter-app constants (split from `ui.py`)."""
from ui_base import (
    CONFIG_PATH,
    MODEL_PATH,
    TOKENIZER_PATH,
    load_config,
)

# Load configuration (kept identical to the former ui.py module global)
cfg = load_config(CONFIG_PATH)

VOCAB_SIZE = cfg["model"]["vocab_size"]
CTX = cfg["model"]["ctx"]
EMB = cfg["model"]["emb"]
HIDDEN = cfg["model"]["hidden"]
