# ScratchpadLM Codebase Structure

> **Project name**: ScratchpadLM (public-facing)  
> **Internal package**: `tinylm` / `TinyLM` (unchanged)
>
> Quick reference for AI agents. Read this instead of reading all source files.

---

## File Overview

| File | Purpose | Key Classes/Functions |
|------|---------|----------------------|
| `tokenizer.py` | Byte-level BPE tokenizer | `BPETokenizer`, `extend()`, `train()`, `encode()`, `decode()`, `save()`, `load()` |
| `ai.py` | Backward-compatible shim re-exporting the `tinylm/` package | Same public API as before (`Value`, `TinyLM`, `Router`, `train()`, `generate()`, …) |
| `tinylm/autograd.py` | Minimal scalar autograd | `Value`, `dot()`, `matvec_dot()`, `softmax_cross_entropy()` |
| `tinylm/serialization.py` | 4-bit model I/O, vocab growth, param counting | `save_model()`, `load_model()`, `sync_vocab()`, `count_params()`, `enforce_param_limit()`, `load_corpus()` |
| `tinylm/models_tiny.py` | TinyLM model | `TinyLM` |
| `tinylm/transformer.py` | Transformer primitives | `layer_norm()`, `attention()`, `feed_forward()`, `TransformerBlock` (+ fast variants) |
| `tinylm/router.py` | Router model (300K params) | `RouterConfig`, `Router` |
| `tinylm/runtime.py` | Multi-model runtime | `ModelRuntime` |
| `tinylm/specialized.py` | Specialized model base (≤100K params) | `SpecializedLM`, `SpecializedConfig`, `create_specialized_model()`, `get_specialized_presets()` |
| `tinylm/router_data.py` | Router training data + pretraining | `RouterTrainingExample`, `encode_router_example()`, `train_router()`, `generate_router_training_data()` |
| `tinylm/data_clean.py` | Training data cleaning / classification | `normalize_text()`, `clean_training_data()`, `classify_by_domain()` |
| `tinylm/sched.py` | LR schedule, clipping, auto-adjust, eval, early stopping | `LR_Scheduler`, `clip_grads()`, `AutoAdjust`, `evaluate()`, `EarlyStopping` |
| `tinylm/checkpoint.py` | Checkpoints + flat param helpers | `Checkpoint`, `save_checkpoint()`, `load_checkpoint()`, `resume_from_checkpoint()`, `_flat_params()` |
| `tinylm/training.py` | Sequential training (Python + C kernel paths) | `train()`, `_train_c()` |
| `tinylm/parallel.py` | Data-parallel training | `train_parallel()`, `_WorkerState` |
| `tinylm/generation.py` | Text generation + sampling | `generate()`, `generate_with_preview()`, `fast_logits()`, `_sample_next_token()` |
| `accel.py` | Optional C training kernel via ctypes (loads `libtinylm_accel.so` from `build.sh`) | `AVAILABLE`, `batch_loss_grads()`, `train_step()`, `CAccelError` |
| `csrc/train_kernel.c` | C source for fused sampled-softmax fwd/bwd + SGD | `tl_batch_loss_grads()`, `tl_train_step()` |
| `config.py` | Config system (YAML/JSON, validation, defaults) | `Config`, `load_yaml()`, `model_config()`, `training_config()`, `merge_configs()` |
| `registry.py` | Model/Task registries, Tokenizer singleton | `ModelRegistry`, `TaskRegistry`, `TokenizerManager`, `TaskTemplate` |
| `model_mgmt.py` | Dynamic model management | `ModelSpec`, `ModelCatalog`, `ModelLoader`, `ModelCreator`, `ModelUpdater`, `MODEL_CATALOG` |
| `ui.py` | Tkinter GUI shim (stdlib fallback): `App` composed from `uitk/` mixins + entry point | `App`, background thread via `queue.Queue`, `_append_history()` with role-based styling |
| `uitk/` | Tkinter implementation (one mixin module per tab/feature) | `core` (init/backend), `shell` + `tab_*` (6 tabs), `history`, `worker`, `act_*` (chat/training/admin/settings), `result`, `info`, `const` |
| `ui_qt.py` | PySide6 (Qt6) GUI shim — primary UI toolkit: `MainWindow`, `run_qt_app()` + entry point | `MainWindow`, `AssistantBubble` (thumbs up/down → RatingCollector), `run_qt_app()` |
| `uiqt/` | PySide6 implementation (one mixin module per tab/feature) | `widgets` (bubbles), `core`, `shell` + `tab_*` (6 tabs), `chat`, `worker`, `logging`, `result`, `act_*` |
| `ui_base.py` | Shared UI layer (no toolkit imports) | `detect_toolkit()`, `qt_platform_check()` (QApplication preflight — Qt aborts instead of raising on broken platform plugins), `run_gui()`, `PALETTE`, `qss_theme()`, `BackgroundWorker`, `write_config_yaml()`, `parse_bool()`, `get_available_models()` |
| `main.py` | CLI + GUI entry point (`tinylm` subcommands, `run_gui`) | `run_gui()`, `run_cli()` |
| `AGENTS.md` | Agent guidance (this project's rules) | — |
| `ROADMAP.md` | Task list with checkboxes | — |
| `STRUCTURE.md` | This file | — |

---

## Module Dependencies

```
uitk/ (Tkinter mixins behind ui.App) / uiqt/ (Qt mixins behind ui_qt.MainWindow)
  ├─ uitk/core.py + uiqt/core.py (init, backend model loading/switching)
  ├─ uitk/shell.py + uiqt/shell.py (tab shell) + uitk/tab_* + uiqt/tab_* (per-tab builders)
  ├─ uitk/history.py + uiqt/chat.py + uiqt/logging.py (conversation, rating, logs)
  ├─ uitk/worker.py + uiqt/worker.py (background threads, queue draining)
  ├─ uitk/act_* + uiqt/act_* (chat/training/admin/settings actions)
  ├─ uitk/result.py + uiqt/result.py (worker result dispatch)
  └─ uitk/const.py (shared Tk constants) / uiqt/widgets.py (chat bubbles)

ui.py (Tkinter shim: App) / ui_qt.py (Qt shim: MainWindow, run_qt_app)
  ├─ ui_base.py (shared layer: detect_toolkit, run_gui, PALETTE, BackgroundWorker, helpers)
  ├─ tokenizer.py (BPETokenizer)
  ├─ ai.py (TinyLM, Router, SpecializedLM, train, generate, load_model, save_model, sync_vocab, train_parallel)
  ├─ config.py (Config)
  └─ registry.py (TokenizerManager, MODEL_REGISTRY, TASK_REGISTRY)

ai.py (shim → tinylm/*)
  ├─ tinylm/autograd.py (Value, dot, matvec_dot, softmax_cross_entropy)
  ├─ tinylm/serialization.py (save/load_model, sync_vocab, count_params)
  ├─ tinylm/models_tiny.py (TinyLM) + tinylm/transformer.py (TransformerBlock)
  ├─ tinylm/router.py (Router) + tinylm/runtime.py (ModelRuntime)
  ├─ tinylm/specialized.py (SpecializedLM) + tinylm/router_data.py (train_router)
  ├─ tinylm/data_clean.py (clean_training_data, classify_by_domain)
  ├─ tinylm/sched.py (LR_Scheduler, AutoAdjust, EarlyStopping) + tinylm/checkpoint.py
  ├─ tinylm/training.py (train, _train_c) + tinylm/parallel.py (train_parallel)
  ├─ tinylm/generation.py (generate, fast_logits, _sample_next_token)
  ├─ tokenizer.py (BPETokenizer)  ← only in __main__ demo
  └─ accel.py (optional C kernel; falls back to pure Python when missing)

registry.py
  ├─ tokenizer.py (BPETokenizer)
  ├─ ai.py (TinyLM, Router, SpecializedLM)  ← only in __main__ demo
  └─ config.py (Config)  ← for future config-driven registration

model_mgmt.py
  ├─ ai.py (TinyLM, Router, SpecializedLM, save_model, load_model, count_params)
  ├─ registry.py (TokenizerManager)
  └─ config.py (Config)

tokenizer.py
  └─ (stdlib only: re, collections, multiprocessing)

config.py
  └─ (stdlib only: json, pathlib, typing)
```

---

## Key Classes

### `BPETokenizer` (tokenizer.py)
- **Vocab**: `self.vocab: Dict[int, bytes]` — token ID → byte sequence
- **Merges**: `self.merges: Dict[Tuple[int, int], int]` — (token_a, token_b) → new_id
- **Critical methods**:
  - `extend(text, vocab_size, min_frequency=2)` — grows vocab *without renumbering existing IDs*
  - `train(text, vocab_size, ...)` — initial training from scratch
  - `encode(text) → List[int]`, `decode(ids) → str`
  - `save(path)`, `load(path)` — persists merges only (vocab rebuilt on load)

### `Value` (tinylm/autograd.py)
- Minimal autograd scalar: `data`, `grad`, `_prev`, `_back`
- Ops: `+`, `*`, `-`, `/`, `**`, `exp()`, `log()`, `tanh()`, `backward()`
- **Rule**: Never send `Value` objects to subprocesses (use floats only)

### `TinyLM` (tinylm/models_tiny.py)
- **Architecture**: Embedding (C) → concat context → tanh hidden (W1, b1) → output (W2, b2)
- **Params** (in order):
  - `C`: `[vocab_size, emb]` — token embeddings
  - `W1`: `[hidden, ctx * emb]`, `b1`: `[hidden]`
  - `W2`: `[vocab_size, hidden]`, `b2`: `[vocab_size]`
- **Methods**:
  - `_embed_context(context) → List[Value]`
  - `logits(context) → List[Value]`
  - `loss(context, target) → Value`
  - `sampled_loss(context, target, n_negatives, vocab_size) → Value` — negative sampling

### `Router` (tinylm/router.py)
- **Architecture**: Token embeddings + positional embeddings → 2 TransformerBlocks → LM head
- **Config**: `RouterConfig(vocab_size, ctx, emb, hidden, n_layers=2, d_ff=512)`
- **Params**: ~300K (embeddings + 2 blocks + output head)
- **Methods**:
  - `_embed_context(context) → List[Value]`
  - `logits(context) → List[Value]`
  - `sampled_loss(context, target, n_negatives, vocab_size) → Value`
  - `fast_logits(context) → List[float]` (inference)
  - `split(prompt, tokenizer, max_subtasks=4, catalog=None) → List[dict]` — task decomposition with optional catalog-based routing

### `SpecializedLM` (tinylm/specialized.py)
- **Architecture**: Token embeddings + positional embeddings → 1 TransformerBlock → LM head
- **Config**: `SpecializedConfig(vocab_size, ctx, emb, hidden, n_layers=1, d_ff=128, model_type)`
- **Params**: ≤100K (enforced at init via `enforce_param_limit`)
- **Methods**:
  - `_embed_context(context) → List[Value]`
  - `logits(context) → List[Value]`
  - `sampled_loss(context, target, n_negatives, vocab_size) → Value`
  - `fast_logits(context) → List[float]` (inference)
- **Factory**: `create_specialized_model(config)`, `get_specialized_presets()` — 8 preset configs (code, reason, prose, embed, sql, docs, shell, data)

### `Config` (config.py)
- Loads YAML/JSON, merges with `DEFAULTS`, validates against `SCHEMA`
- `Config.from_file(path)`, `Config.from_dict(data)`
- Helpers: `model_config()`, `training_config()`, `generation_config()`, `tokenizer_config()`, `merge_configs()`

### `ModelRegistry` (registry.py)
- `register(name, factory, **metadata)` — register model factory
- `get(name) → factory`, `create(name, *args, **kwargs) → model`
- `list() → List[str]`, `metadata(name) → Dict`, `has(name) → bool`, `remove(name) → bool`
- Global instance: `MODEL_REGISTRY`

### `TaskRegistry` (registry.py)
- `register(TaskTemplate)` — register task template
- `get(name) → TaskTemplate`, `list() → List[str]`, `has(name)`, `remove(name)`
- Global instance: `TASK_REGISTRY`

### `TaskTemplate` (registry.py)
- Dataclass: `name`, `model_name`, `prompt_template`, `output_parser`, `description`, `parameters`

### `TokenizerManager` (registry.py)
- Singleton: `TokenizerManager()` always returns same instance
- `tokenizer` property → shared `BPETokenizer`
- `load(path)`, `save(path)`, `extend(text, vocab_size, min_frequency)`, `train(text, vocab_size, ...)`
- `encode(text)`, `decode(ids)`, `vocab_size`, `vocab`, `merges`
- Global instance: `TOKENIZER_MANAGER`

### `App` (ui.py)
- **Layout**: `ttk.Notebook` with 5 tabs — **Chat** (model selector + conversation history + prompt input + quick gen settings), **Training** (training data ScrolledText + steps entry + Train button + training monitor log), **Model Info** (inspector labels + tensor details popup + refresh), **Admin** (catalog Listbox + create model name/domains + delete selected + refresh catalog), **Settings** (generation/training/tokenizer/model config entries + Save Config → writes `config.yaml` via `_write_config_yaml()`)
- **Threading**: `_run_in_background(label, func, *args)` → daemon thread → `queue.Queue` → `root.after(100, _poll_queue)`
- **History**: `_append_history(role, text)` with tags: "user" (blue bold), "assistant" (green bold), "system" (gray italic), "error" (red bold)
- **Training monitor**: `_log_train(message)` — timestamped lines in Training tab log; `ttk.Progressbar` (determinate) updated via `("progress", "Training", (done, total, loss))` queue messages from `progress_cb`
- **Vocab display**: Live label showing current tokenizer vocab size
- **Generation settings**: Configurable n_tokens (default 60) and temperature (default 0.8)

### Serialization & Model Introspection (tinylm/serialization.py)
- `save_model(model, path)` — saves any model with `get_config()` and `get_tensors()` to 4-bit binary
- `load_model(path, model_factory=None)` — loads model, uses registry fallback by `config['arch']`
- `count_params(model)` — returns total parameter count from `get_tensors()`
- `enforce_param_limit(model, max_params, model_name)` — raises if model exceeds limit
- Models must implement:
  - `get_config() → dict` with `"arch"` key
  - `get_tensors() → Dict[str, List[List[Value]]]`

### Dynamic Model Management (model_mgmt.py)

#### `ModelSpec` (dataclass)
- Fields: `name`, `path`, `domain_keywords`, `param_count`, `config`, `created_at`, `metrics`
- `to_dict()`, `from_dict()` for JSON serialization

#### `ModelCatalog`
- `add(spec)`, `remove(name)`, `get(name)`, `list()`, `find_by_domain(keywords)`, `update(name, **kwargs)`, `update_metrics(name, metrics)`
- Persists to JSON file (`models/catalog.json`)
- Global instance: `MODEL_CATALOG`

#### `ModelLoader`
- `load_model(name, force_reload=False)` — loads from catalog, injects shared tokenizer, validates param limit
- `unload(name)`, `unload_all()`, `is_loaded(name)`, `get_loaded(name)`

#### `ModelCreator`
- `create_model(name, domain_keywords, base_config=None, model_factory=None)` — creates and registers new model
- Saves model to `models/<name>.tlm`, adds spec to catalog

#### `ModelUpdater`
- `update_model(name, new_config)` — hot-swaps config, preserves weights where shapes match
- Uses `get_tensors()` to copy matching weights

---

## Training Flow

```
train(model, ids, steps, lr, batch_size, n_negatives, progress_cb)
  └─ for each step:
       ├─ sample batch_size random windows
       ├─ model.sampled_loss() for each (context, target)
       ├─ average losses (sum + scale)
       ├─ backward()
       └─ in-place SGD: p.data -= rate * p.grad

train_parallel(model, ids, steps, lr, batch_size, n_negatives, workers)
  └─ Pool(workers) with persistent _WorkerState.model
     ├─ _init_worker(flat_params, cfg, vocab_size, n_negatives)
     ├─ each step: split batch → pool.map(_worker_grads_v2)
     ├─ sum gradients across workers
     └─ single SGD update on main process
```

---

## Generation Flow

```
generate(model, tok, prompt, n_tokens, temperature)
  └─ ids = tok.encode(prompt)
     ├─ for n_tokens:
     │   ├─ window = last ctx tokens (left-pad with 0)
     │   ├─ fast_logits(model, window) → List[float] (no autograd)
     │   ├─ softmax with temperature
     │   └─ sample next_id
     └─ return tok.decode(ids)
```

---

## 4-bit Quantization (ai.py)

```
save_model(model, path)
  └─ binary format:
       MAGIC "TLM4"
       vocab_size, ctx, emb, hidden (4× uint32)
       For each tensor (C, W1, b1, W2, b2):
         rows, cols (2× uint32)
         scale per row (float32)
         packed 4-bit values (2 per byte)

load_model(path) → TinyLM
  └─ rebuilds model with saved arch, dequantizes into Value.data
```

---

## Vocab Growth Protocol

**NEVER renumber token IDs.** When tokenizer vocab grows:
1. `tok.extend(new_text, vocab_size=NEW_SIZE)` — adds merges, keeps old IDs
2. `tok.save(TOKENIZER_PATH)`
3. `sync_vocab(model, len(tok.vocab))` — grows C, W2, b2 in place, updates `model.params`

---

## UI Threading Model (ui.py / ui_qt.py)

- **Main thread**: GUI toolkit only (Tkinter or Qt)
- **Background**: `threading.Thread` → `queue.Queue` → main-thread timer
  - Tkinter (`ui.py`): `_run_in_background(label, func, *args)` → `root.after(100, _poll_queue)`
  - PySide6 (`ui_qt.py`): `ui_base.BackgroundWorker.submit(label, func, *args)` → `QTimer` (100 ms) → `_poll_worker`
- Both dispatch `("ok"/"error"/"progress"/"log"/"stream_token", label, payload)` messages to `_handle_result` / progress handlers on the main thread
- Workers never touch widgets — they only enqueue (e.g. `worker.post(...)`)
- **UI Layout**: 6 tabs — Chat, Training, Model Info, Admin, Settings, Logs
  - Tkinter: `ttk.Notebook` + scrollable tabs
  - PySide6: `QTabWidget`; Chat tab = sidebar (session list) + bubble chat area + input/send
- **History styling**: colors from shared `ui_base.PALETTE`
  - Tkinter: text tags ("user" blue bold, "assistant" green bold, "system" gray italic, "error" red bold)
  - PySide6: `MessageBubble`/`AssistantBubble` widgets; assistant bubbles have thumbs up/down → `RatingCollector`

---

## Multiprocessing Rules (ai.py)

- All `Pool` code under `if __name__ == "__main__":`
- Workers receive **flat floats** (`_flat_params`), never `Value` objects
- `_WorkerState` holds model instance per worker (persistent across steps)
- `_set_flat_params(model, flat)` copies floats into `Value.data` in place

---

## Saved Artifacts

| File | Description | Do Not Delete |
|------|-------------|---------------|
| `tinyLM` | 4-bit quantized model weights | ✅ |
| `bpe_merges.txt` | Tokenizer merges (one per line: `a b idx`) | ✅ |

---

## Constraints Checklist (from AGENTS.md)

- [ ] Stdlib only (no NumPy, PyTorch)
- [ ] Never renumber token IDs
- [ ] Never recreate model on vocab growth → use `sync_vocab`
- [ ] Tkinter calls only on main thread
- [ ] Multiprocessing under `if __name__ == "__main__":`
- [ ] Send only floats to workers, never `Value`
- [ ] In-place updates: `p.data -= ...`
- [ ] All models 4-bit quantized via `save_model`/`load_model`
- [ ] Max 100K params per specialized model (enforced at init)
- [ ] Router: 300K params, pretrained, frozen by default
- [ ] Single shared tokenizer
- [ ] No fixed context length — dynamic positional embeddings
- [ ] Variable model count — register/unregister at runtime

---

## Entry Points

```bash
# GUI (auto-detects PySide6 vs Tkinter fallback)
python3 ui.py

# CLI (router train, model mgmt, autotrain, generate, train, serve)
python3 main.py --cli

# Train from scratch (in ai.py __main__)
python3 ai.py
```

---

## Next Integration Points

1. **Router-Catalog integration**: Router reads catalog `domain_keywords` for routing decisions (partially done in `Router.split()`)
2. **Multi-Model Runtime**: Load router + specialized models, route prompts collaboratively
3. **Task System**: Task definitions, planner, executor, built-in tasks
4. **CLI / API**: `tinylm` entry point with subcommands (train, generate, serve, model mgmt)
5. **Documentation**: README, CONFIG, HARNESS, MODELS, API docs

---

## Common Patterns

```python
# Grow vocab safely
tok.extend(new_text, vocab_size=2000)
tok.save("bpe_merges.txt")
sync_vocab(model, len(tok.vocab))  # model grows in place

# Train with config
cfg = Config.from_file("config.yaml")
train(model, ids, steps=cfg["training"]["steps"], lr=cfg["training"]["lr"], ...)

# Save/load model (4-bit, works for any architecture)
save_model(model, "my_model.tlm")
loaded = load_model("my_model.tlm")

# Count parameters
n = count_params(model)
enforce_param_limit(model, 100_000, "specialized-model")

# Model registry
from registry import register_model, create_model, list_models
register_model("my-model", lambda: TinyLM(vocab_size=1000, ctx=50, emb=64, hidden=128),
               description="My model", param_count=75000, domain="custom")
model = create_model("my-model")

# Task registry
from registry import register_task, get_task, TaskTemplate
register_task(TaskTemplate(name="summarize", model_name="tiny-prose",
                           prompt_template="Summarize: {text}"))
task = get_task("summarize")

# Shared tokenizer
from registry import TOKENIZER_MANAGER, get_tokenizer
TOKENIZER_MANAGER.load("bpe_merges.txt")
tok = get_tokenizer()
ids = tok.encode("hello")

# Dynamic model management
from model_mgmt import ModelCatalog, ModelCreator, ModelLoader, ModelUpdater

catalog = ModelCatalog("models/catalog.json")
creator = ModelCreator(catalog, "models", tokenizer_manager=TOKENIZER_MANAGER)
spec = creator.create_model("tiny-code", ["code", "python"],
                            base_config={"ctx": 100, "emb": 64, "hidden": 128})

loader = ModelLoader(catalog, tokenizer_manager=TOKENIZER_MANAGER)
model = loader.load_model("tiny-code")

updater = ModelUpdater(catalog, loader)
updater.update_model("tiny-code", {"hidden": 256})
```