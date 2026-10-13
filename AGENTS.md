# AGENTS.md — ScratchpadLM (`tinylm` / `TinyLM`)

Workflow: read STRUCTURE.md → ROADMAP.md → do ONE task → `python3 test_all.py` (must pass) → check the ROADMAP box (+ add discovered tasks) → `git add -A && git commit -m "Task: <desc>"`.
Read only code needed for the task, and only after picking it. Keep replies short and factual; cite code as `path:line`.

## Files

| Area | Location |
|------|----------|
| Core | `tinylm/` (`autograd`, `models_tiny`, `transformer`, `router`, `runtime`, `specialized`, `router_data`, `data_clean`, `sched`, `checkpoint`, `training`, `parallel`, `generation`); demo via `python3 -m tinylm` |
| Tokenizer | `tokenizer.py` — byte-level BPE; `extend()` never renumbers IDs |
| Config | `config.py` — YAML/JSON + `DEFAULTS`/`SCHEMA` validation, stdlib only |
| Registries | `registry.py` — `MODEL_REGISTRY`, `TASK_REGISTRY`, `TOKENIZER_MANAGER` singleton |
| Model mgmt | `model_mgmt.py` — `ModelCatalog` (JSON), `ModelLoader/Creator/Updater` |
| C accel | `accel.py` + `csrc/train_kernel.c` (build: `./build.sh` → `libtinylm_accel.so`) |
| UI | `uitk/app.py` (Tk fallback) + `uiqt/window.py` (PySide6 primary) + `ui_base.py` (shared layer); boot via `main.py`, toolkit override `SCRATCHPADLM_UI=pyside6\|tkinter` |
| Entry | `main.py` — CLI (`tinylm` subcommands) + GUI boot |
| Tests | `test_all.py` — run after EVERY change |

Never delete: `tinyLM` (weights), `bpe_merges.txt` (merges).

## Hard Constraints

Stdlib only. Never renumber token IDs (`extend()` keeps IDs). Never recreate a model on vocab growth (use `sync_vocab`). In-place param updates only (`p.data -= rate * p.grad`). Send only floats to workers, never `Value`. Multiprocessing under `if __name__ == "__main__"`. Tk/Qt calls on main thread only (workers report via queue + timer). All models 4-bit via `save_model`/`load_model`. Specialized models ≤100K params (enforced at init); router 300K, pretrained, frozen by default. Single shared tokenizer. No fixed context length; variable model count; no folder corpus loader in UI.

## Protocols

- **Vocab growth**: `tok.extend(text, NEW_SIZE)` → `tok.save()` → `sync_vocab(model, len(tok.vocab))`.
- **`train()`** (`tinylm/training.py`): sampled softmax (target + `n_negatives`); loss averaged over `batch_size`; `cosine_warmup|linear|constant` LR with warmup → `min_lr`; global-norm `grad_clip` (0 off); `val_split`/`eval_every` perplexity; early stopping (patience, restores best); JSON checkpoints (`checkpoint_every`, `checkpoint_final.json`); `auto_adjust` (LR cut by `auto_factor` on plateaus + clip relaxation); `progress_cb(done,total,loss,val_loss,lr)`; `use_c` forces C kernel (auto when no val/early-stop/checkpoint/cancel). `cancel_callback` stops early → keeps best weights iff `best_loss < initial_loss` (else restores initial); returns `{cancelled, initial_loss, best_loss, final_loss, steps_done}`. Same contract for `_train_c()` and `train_parallel()` (parallel: persistent `_WorkerState`, flat floats in, summed grads, one SGD step).
- **`generate()`** (`tinylm/generation.py`): `max_tokens` cap, `min_tokens` floor, `stop_token_ids` / `stop_strings` autostop, temperature/top-k/top-p/repetition-penalty; `stream_callback` returning `False` stops early; `fast_logits()` = plain-float forward pass.
- **Checkpoints**: `save/load_checkpoint()` (JSON), `resume_from_checkpoint()` restores `Value.data` in place.
- **UI**: `uitk/app.py: App` (Tk: thread → `queue.Queue` → `root.after(100)`), `uiqt/window.py: MainWindow` (Qt: `BackgroundWorker` → `QTimer` 100ms). Training/chat/admin/vocab/config logs go to Logs/Training monitor only — chat shows conversation + model-switch notices + generation errors.

## Recipes

- Config: default in `Config.DEFAULTS` + rule in `Config.SCHEMA`; read via `cfg.get(sec, key)`.
- Model: implement `get_config()` (with `arch`) + `get_tensors()`; count with `count_params()`; register factory in `ModelRegistry`.
- Tokenizer: share `TOKENIZER_MANAGER`; on growth use `extend()`, never `train()`.
- Catalog: `ModelCatalog` JSON file; create/load/hot-swap via Creator/Loader/Updater.
- Tasks: `TaskTemplate` + `TaskRegistry`; debug autograd by walking `Value._prev` after `backward()`.
- Full patterns: see STRUCTURE.md.

## Commands

```bash
python3 test_all.py                 # all (~seconds, must pass before commit)
python3 test_all.py <filter>        # e.g. tokenizer, "c kernel", train_parallel
python3 main.py [--cli]             # GUI (auto toolkit) / CLI
python3 -m tinylm                   # train-from-scratch demo
python3 config.py tokenizer.py      # self-tests
./build.sh                          # rebuild C kernel after editing csrc/train_kernel.c
```

## Anti-Patterns

`TinyLM(...)` after vocab growth → `sync_vocab`; `tok.train(...)` on existing vocab → `tok.extend(...)`; `p = Value(...)` → `p.data -= ...`; `pool.map` with models → flat floats; widget calls from threads → queue + main-thread timer; `import numpy/torch` → stdlib only.
