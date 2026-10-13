# AGENTS.md

Guidance for AI coding agents working in **ScratchpadLM** (internal package: `tinylm` / `TinyLM`).
After reading this document, read **STRUCTURE.md** for codebase overview, then **ROADMAP.md**, select a task, do the task, check if something can be added to the roadmap, mark it as finished, and make a git commit.
Do only one task.
**Only read the parts of the codebase that are needed for the selected task, and only after selecting a task.**

---

## File Overview

| Component | File | Description |
|-----------|------|-------------|
| Tokenizer | `tokenizer.py` | Byte-level BPE. `extend()` adds tokens without changing existing IDs. |
| Model & Autograd | `tinylm/` package (`autograd`, `models_tiny`, `router`, `specialized`, `training`, `parallel`, `generation`, `serialization`, …) | `Value` (autograd), `TinyLM`, `Router`, `SpecializedLM`, training (SGD + sampled softmax), generation, 4-bit save/load. Training extras: `LR_Scheduler`, `clip_grads`, `evaluate`, `EarlyStopping`, `Checkpoint`, `AutoAdjust`. Router training: `train_router()`, `generate_router_training_data()`. Import via `from tinylm import ...`. |
| Config | `config.py` | YAML/JSON config with validation, defaults, merging. Stdlib only. |
| Registries | `registry.py` | `ModelRegistry`, `TaskRegistry`, `TokenizerManager` (singleton). |
| Model Management | `model_mgmt.py` | `ModelSpec`, `ModelCatalog`, `ModelLoader`, `ModelCreator`, `ModelUpdater`. |
| C Acceleration | `accel.py` + `csrc/train_kernel.c` | Optional fused C kernel (ctypes). Supports LR-per-step and gradient clipping (`max_norm`). Built via `./build.sh`. |
| Tests | `test_all.py` | Fast unit test suite (stdlib only). Run `python3 test_all.py` after every change. |
| UI | `uitk/` + `uiqt/` + `ui_base.py` | Tkinter app (`uitk/app.py: App`, stdlib fallback), PySide6 app (`uiqt/window.py: MainWindow`, primary), shared layer (toolkit detection, theme palette, background worker, config helpers). Launch via `main.py` / `ui_base.run_gui()` — PySide6 preferred, Tkinter fallback (`SCRATCHPADLM_UI=pyside6\|tkinter` to force) |
| Entry Point | `main.py` | CLI (`tinylm` subcommands) + GUI boot. Train-from-scratch demo: `python3 -m tinylm`. |

**Saved artifacts** (keep together, do not delete):
- `tinyLM` — 4-bit quantized model weights
- `bpe_merges.txt` — Tokenizer merges

---

## Hard Constraints (Must Follow)

| Rule | Details |
|------|---------|
| **Stdlib only** | No NumPy, PyTorch, or any external packages. |
| **Never renumber token IDs** | `extend()` preserves existing IDs; `sync_vocab` grows model in place. |
| **Never recreate model on vocab growth** | Use `sync_vocab(model, new_vocab_size)`. |
| **Tkinter only on main thread** | Background work reports via `queue.Queue` + `root.after()`. |
| **Multiprocessing under `if __name__ == "__main__":`** | Required for Windows/macOS compatibility. |
| **Send only floats to workers** | Never send `Value` objects to subprocesses. |
| **In-place updates only** | `p.data -= rate * p.grad`, never replace `Value` objects. |
| **4-bit quantized save/load** | All models use `save_model()` / `load_model()`. |
| **Max 100K params per specialized model** | Enforced at init (future: `count_params()`). |
| **Router: 300K params, pretrained, frozen by default** | Separate from specialized models. |
| **Single shared tokenizer** | All models use the same vocab. |
| **No fixed context length** | Dynamic positional embeddings (future). |
| **Variable model count** | Register/unregister at runtime (future). |
| **No folder corpus loader in UI** | Only direct text input. |

---

## Key Protocols

### Vocab Growth (Critical)
```python
# 1. Extend tokenizer (keeps existing IDs)
tok.extend(new_text, vocab_size=NEW_SIZE)
tok.save(TOKENIZER_PATH)

# 2. Grow model embeddings in place
sync_vocab(model, len(tok.vocab))  # Updates C, W2, b2, model.params
```

### Training (Sequential)
```python
train(model, ids, steps=2000, lr=0.05, batch_size=16, n_negatives=256, progress_cb=None,
      lr_scheduler="cosine_warmup", warmup_steps=100, min_lr=0.0,
      grad_clip=1.0, val_ids=None, val_split=0.0, eval_every=50,
      early_stopping=True, patience=5, min_delta=1e-4,
      checkpoint_every=100, checkpoint_dir="checkpoints",
      auto_adjust=True, auto_factor=0.5, auto_patience=10, auto_clip_relax=True)
```
- Sampled softmax: target + `n_negatives` random tokens
- Loss averaged over `batch_size` windows per step
- `lr_scheduler`: `cosine_warmup` | `linear` | `constant`; warmup applies first, then decay to `min_lr`
- `grad_clip`: global-norm gradient clipping (0 disables)
- `val_split`: holds out a fraction of `ids` for perplexity checks every `eval_every` steps
- `early_stopping`: stops after `patience` non-improving evals and restores best weights
- `checkpoint_every`: writes JSON checkpoints (`save_checkpoint`) to `checkpoint_dir`, plus `checkpoint_final.json`
- `auto_adjust`: **automatic adjustment** — reduces LR by `auto_factor` on loss plateaus (after `warmup_steps`, patience `auto_patience`), and relaxes the grad-clip limit if clipping fires on consecutive steps (`auto_clip_relax`)
- `progress_cb(done, total, loss, val_loss=None, lr=None)` called after every step (optional)
- `use_c` forces the C kernel on/off; default: on when `accel.AVAILABLE`
- C kernel (`./build.sh` → `libtinylm_accel.so`, loaded via ctypes in `accel.py`): fused fwd/bwd/SGD on flat floats, supports LR-per-step + `max_norm` gradient clipping; used automatically when validation/early-stopping/checkpointing are off. Pure-Python autograd path is the fallback

### Training (Parallel)
```python
train_parallel(model, ids, steps=2000, lr=0.05, batch_size=16,
               n_negatives=256, workers=4, progress_cb=None, grad_clip=0.0)
```
- Persistent worker pool (`_WorkerState.model` reused)
- Each worker receives flat floats, returns `(loss, grads)`
- Main process sums grads, clips (if `grad_clip > 0`), single SGD update
- `progress_cb(done, total, loss)` runs in main process (safe for UI queues)

### Resume from Checkpoint
```python
from tinylm import load_checkpoint, resume_from_checkpoint
ckpt = load_checkpoint("checkpoints/checkpoint_final.json")
resume_from_checkpoint(model, ckpt)  # in-place restore of Value.data
```

### Generation
```python
generate(model, tok, prompt, n_tokens=60, temperature=0.8)
```
- `fast_logits()` — forward pass on plain floats (no autograd)
- Temperature sampling with softmax

### 4-bit Save/Load
```python
save_model(model, "tinyLM")   # Writes binary TLM4 format
model = load_model("tinyLM")  # Rebuilds TinyLM, dequantizes into Value.data
```

---

## File-Specific Notes

### `tokenizer.py`
- `BPETokenizer.merges`: `(token_a, token_b) → new_id` (ordered by creation)
- `BPETokenizer.vocab`: `id → bytes` (rebuilt on `load()`)
- `extend()`: Re-encodes new text with existing merges, then learns new ones
- `train()`: Parallel word counting, sequential merge learning
- Multiprocessing in `train()` under `if __name__ == "__main__":`

### `tinylm/` package (split from the former monolithic `ai.py`)
- `Value`: Minimal autograd scalar with `__slots__ = ("data", "grad", "_prev", "_back")`
- `matvec_dot()`: Fused matrix-vector + bias as single autograd node
- `softmax_cross_entropy()`: Single node, gradient = probs - one_hot(target)
- `TinyLM.params`: Flat list of all `Value` objects in fixed order
- `_WorkerState`: Module-level namespace for persistent worker model
- `_flat_params()` / `_set_flat_params()`: Convert between `List[Value]` and `List[float]`
- **Router**: `RouterConfig`, `Router` class — 2-layer transformer for task decomposition
- **SpecializedLM**: `SpecializedConfig`, `SpecializedLM` class — 1-layer transformer (≤100K params)
- **Router training**: `train_router()`, `generate_router_training_data()`, `RouterTrainingExample`

### `config.py`
- `Config.from_file(path)`: Loads `.yaml`/`.yml` (custom parser) or `.json`
- `Config.from_dict(data)`: Create from dict, merges with `DEFAULTS`
- `Config.get(section, key, default)`: Safe access
- Schema validation in `_validate()`: type + custom predicate per key
- Helpers: `model_config()`, `training_config()`, `generation_config()`, `tokenizer_config()`, `merge_configs()`

### `uitk/app.py` (Tkinter fallback UI; mixins in `uitk/`)
- `App._run_in_background(label, func, *args)`: Starts daemon thread, puts result on `self.queue`
- `App._poll_queue()`: Called via `root.after(100, ...)`, processes queue on main thread
- `_train_job()`: `tok.extend()` → `tok.save()` → `sync_vocab()` → `train_parallel()`
- `_generate_job()`: `generate(model, tok, prompt)`

### `uiqt/window.py` (PySide6 primary UI; mixins in `uiqt/`)
- `MainWindow`: same 6 tabs as Tkinter (Chat, Training, Model Info, Admin, Settings, Logs) + chat sidebar
- Uses `ui_base.BackgroundWorker` + `QTimer` (100 ms) to drain the queue on the main thread
- Thumbs up/down on assistant bubbles → `RatingCollector.collect(..., rating=5|1)`
- Only touch Qt widgets on the main thread; jobs post messages via `worker.post()`

### `ui_base.py` (shared UI layer)
- `detect_toolkit()` / `run_gui()`: PySide6 preferred, Tkinter fallback; honors `SCRATCHPADLM_UI`
- `qt_platform_check()`: preflight BEFORE `QApplication` — Qt aborts the process (no exception) when a platform plugin can't init (e.g. Qt ≥ 6.5 needs `libxcb-cursor` / `xcb-cursor0` on X11), so this check triggers the Tkinter fallback instead of a crash
- `PALETTE`: shared colors (Tk tag colors + `qss_theme()` Qt stylesheet)
- `BackgroundWorker`: thread-safe `submit()` / `post()` / `drain()` queue
- Shared helpers: `write_config_yaml()`, `parse_bool()`, `get_available_models()`, `load_config()`

---

## Common Tasks

### Add a New Config Option
1. Add default to `Config.DEFAULTS` in `config.py`
2. Add validation rule to `Config.SCHEMA`
3. Use `cfg.get("section", "key")` or `cfg["section"]["key"]` in code

### Add a New Model Type
1. Create new class in `tinylm/` (e.g. `tinylm/models_tiny.py`) inheriting pattern from `TinyLM`
2. Ensure `params` property returns flat list of `Value` in consistent order
3. Add `save_model`/`load_model` support (or extend existing)
4. Register in `ModelRegistry` (roadmap task)

### Register a Model
```python
from registry import register_model, create_model, list_models
from tinylm import TinyLM

register_model("my-model",
    lambda: TinyLM(vocab_size=1000, ctx=50, emb=64, hidden=128),
    description="Custom model", param_count=75000, domain="custom")

model = create_model("my-model")  # Creates new instance
print(list_models())  # ['my-model', ...]
```

### Save/Load Model (4-bit, works for any architecture)
```python
from tinylm import save_model, load_model, count_params, enforce_param_limit

# Model must implement get_config() and get_tensors()
save_model(model, "my_model.tlm")  # Small binary file (~4 bits/param)

loaded = load_model("my_model.tlm")  # Uses registry by config['arch']

# Count parameters
n = count_params(model)
print(f"Params: {n:,}")

# Enforce limits (e.g., ≤100K for specialized, ≤300K for router)
enforce_param_limit(model, 100_000, "specialized-model")  # Raises if exceeded
```

### Dynamic Model Management
```python
from model_mgmt import ModelCatalog, ModelCreator, ModelLoader, ModelUpdater
from registry import TOKENIZER_MANAGER

# Catalog (persistent JSON)
catalog = ModelCatalog("models/catalog.json")

# Create specialized models
creator = ModelCreator(catalog, "models", tokenizer_manager=TOKENIZER_MANAGER)
spec = creator.create_model("tiny-code", ["code", "python"],
                            base_config={"ctx": 100, "emb": 64, "hidden": 128})
# Creates models/tiny-code.tlm, registers in catalog

# Load models (with shared tokenizer injection)
loader = ModelLoader(catalog, tokenizer_manager=TOKENIZER_MANAGER, max_params=100_000)
model = loader.load_model("tiny-code")

# Hot-swap config, preserve weights where shapes match
updater = ModelUpdater(catalog, loader)
updater.update_model("tiny-code", {"hidden": 256})

# Find models by domain
code_models = catalog.find_by_domain(["code", "python"])
```

### Register a Task
```python
from registry import register_task, get_task, list_tasks, TaskTemplate

register_task(TaskTemplate(
    name="summarize",
    model_name="tiny-prose",
    prompt_template="Summarize: {text}",
    description="Text summarization",
    output_parser=lambda x: x.strip()
))

task = get_task("summarize")
prompt = task.prompt_template.format(text="Long article...")
```

### Use Shared Tokenizer
```python
from registry import TOKENIZER_MANAGER, get_tokenizer

# Load existing
TOKENIZER_MANAGER.load("bpe_merges.txt")

# Or get directly
tok = get_tokenizer()
ids = tok.encode("hello world")
text = tok.decode(ids)

# Extend vocab (affects all models using shared tokenizer)
TOKENIZER_MANAGER.extend(new_text, vocab_size=2000)
TOKENIZER_MANAGER.save("bpe_merges.txt")
```

### Extend Tokenizer Safely
```python
# Always use extend, never retrain from scratch on same tokenizer
tok = BPETokenizer()
tok.load("bpe_merges.txt")  # Load existing
tok.extend(new_corpus, vocab_size=LARGER_SIZE)
tok.save("bpe_merges.txt")
sync_vocab(model, len(tok.vocab))
```

### Debug Autograd
```python
# Print computation graph
def print_graph(v, indent=0):
    print("  " * indent + f"Value({v.data:.4f}, grad={v.grad:.4f})")
    for p in v._prev:
        print_graph(p, indent+1)

loss = model.loss(context, target)
loss.backward()
print_graph(loss)
```

---

## Git Workflow

1. Read `STRUCTURE.md` → `ROADMAP.md`
2. Pick an unchecked task from ROADMAP (prefer top of section)
3. Implement following constraints above
4. Test: `python3 test_all.py` (fast, ~seconds — must pass before committing)
5. Update `ROADMAP.md`: check the box, add new tasks if discovered
6. Commit: `git add -A && git commit -m "Task: <description>"`

---

## Testing Commands

```bash
# Fast unit test suite (stdlib only, no pytest needed — run after EVERY change)
python3 test_all.py                    # all tests (~seconds)
python3 test_all.py tokenizer          # filter by name substring
python3 test_all.py "c kernel"         # spaces OK in filter

# Run GUI
python3 main.py

# Test config
python3 config.py

# Test tokenizer
python3 tokenizer.py

# Test model train/generate (tinylm/__main__.py demo)
python3 -m tinylm

# Rebuild C kernel after editing csrc/train_kernel.c
./build.sh                             # -> libtinylm_accel.so (accel.AVAILABLE)
```

---

## Roadmap Sections (from ROADMAP.md)

| Section | Focus |
|---------|-------|
| 0. Foundations | Config, registries, tokenizer singleton, param counter |
| 0a. Control UI | PySide6 dashboard (Tkinter fallback): model mgmt, training, generation, theming (dark/light/system) |
| 0b. Dynamic Model Management | ModelSpec, Catalog, Loader, Creator, Updater, CLI |
| 0c. Auto-Training | RatingCollector, AutoTrainer, feedback loop |
| 1. Router Model | 300K param 2-layer transformer for task decomposition |
| 2. Specialized Model Base | ≤100K param 1-layer transformer factory |
| 3. Default Specialized Models | tiny-code, tiny-reason, tiny-prose, tiny-embed |
| 4. Multi-Model Runtime | Load router + specialists, route prompts |
| 5. Task System | Task definitions, planner, executor, built-ins |
| 6. Collaborative Generation | Scratchpad, coordinator, sync protocol |
| 7. Training Pipeline | Generic train, datasets, LR sched, clipping, validation |
| 8. Tokenizer Enhancements | Word freq cache, streaming corpus, incremental extend |
| 9. Generation Improvements | Top-k, top-p, repetition penalty, unified generate |
| 10. Checkpointing | Checkpoint class, auto-save, resume |
| 11. Harness | .tlm script parser, runtime, assertions, parallel, retry |
| 11a. Skill System | Executable helper scripts: skill manifests, registry, safe runner, CLI/UI, model invocation |
| 12. CLI / API | tinylm entry point, serve, train, eval |
| 13. Testing & Validation | Unit, integration, harness, benchmarks, CI |
| 14. Documentation | README, CONFIG, HARNESS, MODELS, API |

---

## Anti-Patterns to Avoid

| Anti-Pattern | Correct Approach |
|--------------|------------------|
| `model = TinyLM(...)` after vocab growth | `sync_vocab(model, new_vocab_size)` |
| `tok = BPETokenizer(); tok.train(...)` on existing tokenizer | `tok.load(...); tok.extend(...)` |
| `p = Value(new_value)` to update param | `p.data = new_value` or `p.data -= ...` |
| `pool.map(worker, [model, ...])` sending model | Send flat floats, rebuild in worker |
| `threading.Thread(target=tk_widget.update)` | Use `queue` + `root.after()` |
| `import numpy` or `import torch` | Pure Python / stdlib only |

---

## Useful Grep Patterns

```bash
# Find all Value operations
grep -n "class Value\|def __add__\|def __mul__\|def backward" tinylm/autograd.py

# Find training loops
grep -n "def train\|for step in range\|loss.backward" tinylm/training.py tinylm/parallel.py

# Find vocab growth
grep -n "sync_vocab\|\.extend\|vocab_size" tinylm/*.py uitk/*.py uiqt/*.py

# Find multiprocessing
grep -n "Pool\|_WorkerState\|_init_worker\|_worker_grads" tinylm/parallel.py

# Find UI threading
grep -n "queue\|_poll_queue\|_run_in_background\|threading" uitk/worker.py
```
