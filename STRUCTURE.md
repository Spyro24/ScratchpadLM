# ScratchpadLM Codebase Structure

> Project: ScratchpadLM (public) · internal package `tinylm` / `TinyLM`. Read this instead of reading all source files.

## Files

| File(s) | Contents |
|---------|----------|
| `tinylm/autograd.py` | `Value`, `dot`, `matvec_dot`, `softmax_cross_entropy` |
| `tinylm/serialization.py` | `save/load_model` (4-bit), `sync_vocab`, `count_params`, `enforce_param_limit`, `load_corpus` |
| `tinylm/models_tiny.py` + `transformer.py` | `TinyLM`; `TransformerBlock`, `layer_norm`, `attention`, `feed_forward` (+ fast variants) |
| `tinylm/router.py` + `runtime.py` | `RouterConfig`, `Router` (mean-pooled window, catalog-only model names; `split`/`route`/`route_multi`/`decompose`); `ModelRuntime` (lazy-load, TTL, cap-free marker-stop generation) |
| `tinylm/specialized.py` | `SpecializedLM`, `SpecializedConfig`, presets factory (≤100K params) |
| `tinylm/router_data.py` + `data_clean.py` | `RouterTrainingExample`, `train_router` (returns result dict with `model`), training-data corpus utils; `clean_training_data`, `classify_by_domain`, structural markers (`[EOL]`/`[EOS]`/`[EOT]` + yellow render, `mark_and_encode` entry) |
| `tinylm/sched.py` + `checkpoint.py` | `LR_Scheduler`, `clip_grads`, `AutoAdjust`, `evaluate`, `EarlyStopping`; `Checkpoint`, save/load/resume, `_flat_params` |
| `tinylm/training.py` + `parallel.py` | `train`, `_train_c`; `train_parallel`, `train_multicore`, `_WorkerState` |
| `tinylm/stats.py` | `TrainingStats`, `train_with_stats` (monotonic best loss + best snapshots) |
| `tinylm/generation.py` | `generate` (unbounded by default, marker autostop), `generate_with_preview`, `fast_logits`, `_sample_next_token` |
| `tinylm/__main__.py` | `main()` train-from-scratch demo |
| `tokenizer.py` | `BPETokenizer`: `train`/`extend`/`encode`/`decode`/`save`/`load`; merges `(a,b)→id`, vocab `id→bytes` (rebuilt on load) |
| `config.py` | `Config` (YAML/JSON, `DEFAULTS`/`SCHEMA`); `model/training/generation/tokenizer_config()`, `merge_configs()` |
| `registry.py` | `MODEL_REGISTRY`, `TASK_REGISTRY` (+ `TaskTemplate`), `TOKENIZER_MANAGER` singleton |
| `scratchpad.py` | `Scratchpad`, `ThinkEngine`, `TaskPad` (+ `Task`/`TaskType`), `plan()` router→tasks |
| `model_mgmt.py` | `ModelSpec`, `ModelCatalog` (JSON), `ModelLoader/Creator/Updater`, `MODEL_CATALOG` |
| `accel.py` + `csrc/train_kernel.c` | Optional C kernel (`AVAILABLE`, `train_step`, `batch_loss_grads`); `./build.sh` |
| `uitk/app.py` + `uitk/` mixins | Tk `App`: `core` (init/backend), `shell`+`tab_*` (6 tabs), `history`, `worker`, `act_*`, `result`, `info`, `const` |
| `uiqt/window.py` + `uiqt/` mixins | Qt `MainWindow` + `run_qt_app`: `widgets` (bubbles), `core`, `shell`+`tab_*`, `chat`, `worker`, `logging`, `result`, `act_*` |
| `ui_base.py` | `detect_toolkit`/`run_gui` (+ `SCRATCHPADLM_UI` override, `qt_platform_check` preflight), `PALETTE`/`qss_theme`, `BackgroundWorker`, config helpers |
| `main.py` | CLI (`tinylm` subcommands) + GUI boot |
| `opencodeSkill/readpy.py` | File-structure reader skill: `tree`/`treefull`/`class`/`function` for Python files (stdlib `ast`) |
| `opencodeSkill/SKILL.md` | Agent skill index for this repo (name/description + when to use each skill) |
| `test_all.py` | stdlib unit/integration suite |

Never delete: `tinyLM` (weights), `bpe_merges.txt` (merges).

## Dependencies

```
uitk/app.py / uiqt/window.py → ui_base, tokenizer, tinylm, config, registry (+ PySide6 for Qt)
tinylm/* → autograd at bottom; models ← training/generation; training → sched/checkpoint;
           parallel → training/sched/checkpoint; runtime → serialization/generation;
           router_data → training (lazy import). No cycles.
registry → tokenizer (+ tinylm lazily) · model_mgmt → tinylm/registry/config
tokenizer, config: stdlib only
```

## Key Facts

- `Value`: `data/grad/_prev/_back`; ops `+ * - / ** exp log tanh`, `backward()`; never cross process boundary.
- `TinyLM`: embed `C[vocab,emb]` → concat ctx → tanh `W1[hidden,ctx*emb]+b1` → `W2[vocab,hidden]+b2`; `params` flat list in that order; `loss`/`sampled_loss(context,target,…)`.
- `Router`: embeddings + 2 `TransformerBlock` + LM head (~300K); `split`/`route`/`route_multi` use catalog `domain_keywords`.
- `SpecializedLM`: embeddings + 1 block + head (≤100K, enforced); 8 presets via `get_specialized_presets()`.
- Serialization: `TLM4` binary, per-row scale, 2×4-bit/byte; models need `get_config()` (with `arch`) + `get_tensors()`.
- Training step: sample `batch_size` windows → `sampled_loss` → average → `backward()` → clip (mean gradient) → in-place SGD with momentum. Parallel: `Pool` + persistent `_WorkerState`, flat floats only, sum grads, one momentum step in the parent.
- Generation: `window = last ctx ids (0-padded)` → `fast_logits` (floats, vocab-clamped) → temp/top-k/top-p/penalty → sample; autostop on `stop_token_ids`/`stop_strings`/`max_tokens`.
- Vocab growth: `extend` (keeps IDs) → `save` → `sync_vocab` (grows `C/W2/b2` in place).
- UI: main thread owns widgets; jobs post to queue, timer drains (`root.after(100)` / `QTimer` 100ms); messages `ok/error/progress/log/stream_token` → `_handle_result`. Chat shows conversation + model-switch notices + generation errors only.
- Entry: `python3 main.py` (GUI) · `python3 main.py --cli` · `python3 -m tinylm` (demo).

## Common Patterns

```python
tok.extend(text, 2000); tok.save("bpe_merges.txt"); sync_vocab(model, len(tok.vocab))
train(model, ids, steps=cfg["training"]["steps"], lr=cfg["training"]["lr"])
save_model(m, "m.tlm"); m2 = load_model("m.tlm")
register_model("n", lambda: TinyLM(...), description="..", param_count=75000, domain="..")
spec = ModelCreator(cat, "models").create_model("tiny-x", ["kw"], base_config={...})
```
