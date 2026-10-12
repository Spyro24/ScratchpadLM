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
| `ai.py` | Autograd, model, training, generation, 4-bit I/O | `Value`, `TinyLM`, `train()`, `train_parallel()`, `generate()`, `save_model()`, `load_model()`, `sync_vocab()`, `count_params()`, `enforce_param_limit()` |
| `config.py` | Config system (YAML/JSON, validation, defaults) | `Config`, `load_yaml()`, `model_config()`, `training_config()`, `merge_configs()` |
| `registry.py` | Model/Task registries, Tokenizer singleton | `ModelRegistry`, `TaskRegistry`, `TokenizerManager`, `TaskTemplate` |
| `model_mgmt.py` | Dynamic model management | `ModelSpec`, `ModelCatalog`, `ModelLoader`, `ModelCreator`, `ModelUpdater`, `MODEL_CATALOG` |
| `ui.py` | Tkinter GUI with tabbed layout (Chat, Training, Model Info, Admin, Settings) | `App`, background thread via `queue.Queue`, `_append_history()` with role-based styling |
| `main.py` | Legacy demo script (n-gram model) | `tokenize()`, `train()`, `generate()` |
| `AGENTS.md` | Agent guidance (this project's rules) | — |
| `ROADMAP.md` | Task list with checkboxes | — |
| `STRUCTURE.md` | This file | — |

---

## Module Dependencies

```
ui.py
  ├─ tokenizer.py (BPETokenizer)
  ├─ ai.py (TinyLM, train, generate, load_model, save_model, sync_vocab, train_parallel)
  ├─ config.py (Config)
  └─ registry.py (TokenizerManager, MODEL_REGISTRY, TASK_REGISTRY)

ai.py
  ├─ tokenizer.py (BPETokenizer)  ← only in __main__ demo
  └─ config.py (Config)  ← NEW, not yet integrated

registry.py
  ├─ tokenizer.py (BPETokenizer)
  ├─ ai.py (TinyLM)  ← only in __main__ demo
  └─ config.py (Config)  ← for future config-driven registration

model_mgmt.py
  ├─ ai.py (TinyLM, save_model, load_model, count_params)
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

### `Value` (ai.py)
- Minimal autograd scalar: `data`, `grad`, `_prev`, `_back`
- Ops: `+`, `*`, `-`, `/`, `**`, `exp()`, `log()`, `tanh()`, `backward()`
- **Rule**: Never send `Value` objects to subprocesses (use floats only)

### `TinyLM` (ai.py)
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
- **Training monitor**: `_log_train(message)` — timestamped lines in Training tab log
- **Vocab display**: Live label showing current tokenizer vocab size
- **Generation settings**: Configurable n_tokens (default 60) and temperature (default 0.8)

### Serialization & Model Introspection (ai.py)
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
train(model, ids, steps, lr, batch_size, n_negatives)
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

## UI Threading Model (ui.py)

- **Main thread**: Tkinter only
- **Background**: `threading.Thread` → `queue.Queue` → `root.after(100, _poll_queue)`
- `_run_in_background(label, func, *args)` runs `func` in thread, puts `("ok"/"error", label, result)` on queue
- `_poll_queue` dispatches to `_handle_result` on main thread
- **UI Layout**: `ttk.Notebook` tabs — Chat, Training, Model Info, Admin, Settings
- **History styling**: Tags for "user" (blue bold), "assistant" (green bold), "system" (gray italic), "error" (red bold)

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
# GUI
python3 ui.py

# Legacy demo (n-gram)
python3 main.py [corpus.txt]

# Train from scratch (in ai.py __main__)
python3 ai.py
```

---

## Next Integration Points

1. **Wire Config into ui.py**: Replace hardcoded `VOCAB_SIZE, CTX, EMB, HIDDEN` with `Config`
2. **Wire Config into ai.py**: Use `Config` for `train()`, `train_parallel()`, `generate()`
3. **Add ModelRegistry** (roadmap 0): `registry.py` with `register/get/list`
4. **Add TaskRegistry** (roadmap 0): `task_registry.py`
5. **Add TokenizerManager** (roadmap 0): singleton wrapper for `BPETokenizer`

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