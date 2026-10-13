# ScratchpadLM (TinyLM)

**ScratchpadLM** is a lightweight, self-contained language model framework written entirely in Python standard library. It features a modular architecture with a router model for task decomposition and specialized models for different domains.

## Key Features

- **Zero External Dependencies** - Pure Python standard library only (optional C acceleration)
- **4-bit Quantized Models** - Efficient storage and loading via custom TLM4 format
- **Dynamic Vocabulary Growth** - Extend tokenizer without renumbering token IDs
- **Router + Specialized Models** - 2-layer transformer router (300K params) routes to 1-layer specialized models (≤100K params each)
- **Auto-Training from Feedback** - Thumbs up/down ratings trigger automatic fine-tuning
- **Multi-Model Runtime** - Lazy-loading, TTL-based unloading, collaborative generation
- **Dual UI Toolkits** - PySide6 (primary) with Tkinter fallback
- **Model Catalog** - Persistent registry with domain-based discovery
- **Configuration System** - YAML/JSON with validation and defaults

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        User Prompt                            │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                    Router Model (2-layer)                    │
│  • Task decomposition                                        │
│  • Model selection via domain keywords                       │
│  • 300K parameters, pretrained, frozen by default            │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
        ┌────────┬────────┬────────┬────────┬────────┐
        ▼        ▼        ▼        ▼        ▼        ▼
    ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐
    │ code │ │reason│ │prose │ │ sql  │ │docs  │ │ ...  │
    │tiny- │ │tiny- │ │tiny- │ │tiny- │ │tiny- │ │      │
    │ code │ │reason│ │prose │ │ sql  │ │ docs │ │      │
    └──────┘ └──────┘ └──────┘ └──────┘ └──────┘ └──────┘
    ≤100K params each, 1-layer transformer, domain-specialized
```

## Quick Start

### Installation

```bash
git clone https://github.com/yourorg/scratchpadlm
cd scratchpadlm
# Optional: build C acceleration for ~14x faster training
./build.sh
```

### Run GUI

```bash
# PySide6 (preferred)
pip install pyside6
python3 -m ui_qt

# Tkinter (stdlib fallback, no install needed)
python3 -m ui
```

### Command Line Interface

```bash
# Generate text
python3 -m main "Write a Python function to calculate fibonacci"

# Train on data
python3 -m main --cli train --data "your training text here" --steps 100

# Model management
python3 -m main --cli model list
python3 -m main --cli model create tiny-code --domain "code,python"

# Auto-training
python3 -m main --cli autotrain status
python3 -m main --cli autotrain enable
```

### Python API

```python
from ai import TinyLM, generate, train, save_model, load_model
from tokenizer import BPETokenizer

# Create tokenizer and model
tok = BPETokenizer()
tok.train("your corpus text " * 100, vocab_size=1000)

model = TinyLM(vocab_size=len(tok.vocab), ctx=50, emb=64, hidden=128)

# Train
ids = tok.encode("your training text " * 50)
train(model, ids, steps=200, lr=0.05, batch_size=16)

# Generate
text = generate(model, tok, "Once upon a time", n_tokens=60, temperature=0.8)
print(text)

# Save/load (4-bit quantized)
save_model(model, "my_model.tlm")
loaded = load_model("my_model.tlm")
```

## Project Structure

```
scratchpadlm/
├── ai.py              # Core: Value autograd, TinyLM, Router, SpecializedLM, training
├── tokenizer.py       # Byte-level BPE tokenizer with vocab extension
├── config.py          # YAML/JSON config with validation
├── registry.py        # Model/Task registries, TokenizerManager singleton
├── model_mgmt.py      # ModelSpec, Catalog, Loader, Creator, Updater
├── autotrain.py       # RatingCollector, AutoTrainer, ModelSelector, PriorityQueue
├── scratchpad.py      # Scratchpad, ThinkEngine, TaskPad
├── ui.py              # Tkinter GUI (stdlib fallback)
├── ui_qt.py           # PySide6 GUI (primary)
├── ui_base.py         # Shared UI layer (theming, workers, config)
├── accel.py           # C acceleration wrapper (ctypes)
├── csrc/train_kernel.c# Fused C training kernel
├── main.py            # CLI entry point
├── config.yaml        # Configuration file
├── build.sh           # Build C extension
└── test_all.py        # Unit test suite (stdlib only)
```

## Configuration

Create `config.yaml` to customize:

```yaml
model:
  vocab_size: 2000
  ctx: 128
  emb: 64
  hidden: 128

training:
  steps: 1000
  lr: 0.05
  batch_size: 16
  n_negatives: 256
  lr_scheduler: "cosine_warmup"
  warmup_steps: 100
  grad_clip: 1.0
  early_stopping: true

generation:
  n_tokens: 100
  temperature: 0.7
  top_k: 50
  top_p: 0.9
  repetition_penalty: 1.1

ui:
  theme: "system"  # dark, light, system
```

See [CONFIG.md](CONFIG.md) for all options.

## Training Pipeline

1. **Data Input** - Paste text, upload files, or fetch URLs in Training tab
2. **Vocab Extension** - Tokenizer learns new patterns from data
3. **Model Sync** - `sync_vocab()` grows embeddings in-place
4. **Sampled Softmax** - Target + random negatives for efficient training
5. **LR Scheduling** - Cosine warmup, linear, or constant
6. **Auto-Adjust** - Plateau LR reduction + grad clip relaxation
7. **Validation** - Perplexity evaluation with early stopping
8. **Checkpointing** - Auto-save every N steps, resume supported

## Auto-Training System

- **RatingCollector** - Stores thumbs up/down (5/1) in `feedback.jsonl`
- **AutoTrainer** - Background worker batches high-rated feedback, fine-tunes models
- **Deduplication** - Edit-distance based near-duplicate detection
- **Rate Limiting** - Max 1 train per 5 min, 100 steps/day per model
- **Low-Rating Handler** - Flags for review, optional avoidance training

## Model Management

```python
from model_mgmt import ModelCatalog, ModelCreator, ModelLoader
from registry import TOKENIZER_MANAGER

# Catalog persists to models/catalog.json
catalog = ModelCatalog()

# Create specialized model
creator = ModelCreator(catalog, "models", tokenizer_manager=TOKENIZER_MANAGER)
spec = creator.create_model("tiny-rust", ["rust", "systems"],
                            base_config={"ctx": 100, "emb": 64, "hidden": 128})

# Load with shared tokenizer
loader = ModelLoader(catalog, tokenizer_manager=TOKENIZER_MANAGER)
model = loader.load_model("tiny-rust")

# Hot-swap config, preserve weights
updater = ModelUpdater(catalog, loader)
updater.update_model("tiny-rust", {"hidden": 256})
```

## Hard Constraints

| Rule | Enforcement |
|------|-------------|
| Stdlib only | No numpy, torch, or external packages |
| Never renumber token IDs | `extend()` preserves existing IDs |
| Never recreate model on vocab growth | Use `sync_vocab(model, new_vocab_size)` |
| Tkinter only on main thread | Background work via `queue.Queue` + `root.after()` |
| Multiprocessing under `if __name__ == "__main__":` | Required for Windows/macOS |
| In-place updates only | `p.data -= rate * p.grad` |
| 4-bit quantized save/load | All models use `save_model()`/`load_model()` |
| Max 100K params/specialized model | Enforced at init via `enforce_param_limit()` |
| Router: 300K params, pretrained, frozen | Separate from specialized models |
| Single shared tokenizer | All models use `TOKENIZER_MANAGER` |
| No fixed context length | Dynamic positional embeddings |
| Variable model count | Register/unregister at runtime |

## Building C Acceleration

```bash
./build.sh
# Creates libtinylm_accel.so
# Used automatically when accel.AVAILABLE
```

The C kernel fuses forward/backward/SGD with gradient clipping for ~14x speedup.

## Testing

```bash
python3 test_all.py              # All tests (~seconds)
python3 test_all.py tokenizer    # Filter by name
python3 test_all.py "c kernel"   # Spaces OK in filter
```

## License

MIT License - see LICENSE file for details.

## Contributing

1. Follow hard constraints above
2. Run `python3 test_all.py` before committing
3. Update ROADMAP.md when adding features
4. Single task per commit with descriptive message