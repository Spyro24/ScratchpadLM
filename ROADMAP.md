# ScratchpadLM Task List

> Project: ScratchpadLM (public) · internal package `tinylm` / `TinyLM`.

Done to date: fused `matvec_dot`, persistent `train_parallel` workers, O(1)-depth loss accumulation (~2x sequential / ~3x parallel); `tinylm/` + `uitk/` + `uiqt/` module splits with shims removed.

## Backlog (ordered, atomic)

### 0. Foundations
- [x] Config, registries, task registry, tokenizer singleton, 4-bit save/load, param counter, Config wired into UI

### 0a. Control UI (HIGH PRIORITY)
- [x] Prompt interface, tabbed layout (Chat/Training/Tasks/Model Info/Admin/Settings/Logs), PySide6 primary + Tk fallback (`ui_base.py`, auto-detect + `SCRATCHPADLM_UI` override), chat layout (tabs/sidebar/bubbles/input/send/thumbs→RatingCollector→AutoTrainer), theme system (dark/light/system, live switch), auto model naming, train-vs-create advisor, model selector, training monitor, generation controls, tokenizer manager, config editor, log viewer, model inspector, layout polish
- [ ] **UI Toolkit Migration** (parent)
- [ ] **Harness Tab (PySide6)**: privilege selector, root path + browse, mode toggle, privilege help, path validation, status indicator
- [ ] **Image processor**: upload/preview, resize/normalize, vision-model prompts
- [ ] **Training data provider**: file/text/URL/folder-watch input, preview, domain detection, train/val split
- [ ] **Variable batch tokenizer**: mixed-domain batches routed per model, no cross-domain contamination
- [ ] **Control panel**: PySide6 dashboard (Tk fallback)

### 0b. Dynamic Model Management
- [x] `ModelSpec`, `ModelCatalog` (JSON), `ModelLoader`, `ModelCreator`, `ModelUpdater`, catalog CLI, auto-discovery, router↔catalog routing
- [ ] **Harness integration**: `model load/create` in `.tlm` scripts
- [ ] **Model export**: `export_model()` bundles a single model (4-bit `.tlm` weights + tokenizer merges + spec/config JSON) into one portable archive
- [ ] **Model import**: `import_model()` cleanly imports an exported archive — validates format/checksums, registers into the catalog, merges tokens via `extend()` only (never renumbers IDs), `sync_vocab` on affected loaded models

### 0c. Auto-Training
- [x] `RatingCollector` (thresholds 4/5 high, 2/5 low), `ModelSelector`, `AutoTrainer` (batch ≥8, 60s wait, few-shot fine-tune, `autotrain.log`), low-rating handler, dedup, rate limits, per-model metrics
- [ ] **CLI**: `autotrain status/enable/disable/config`
- [ ] **Harness integration**: `autotrain on/off/config`, `rate last <1-5>`

### 1. Router (300K, pretrained, frozen)
- [x] `RouterConfig`, 2-layer transformer, forward + fast inference, sampled-loss training, `(prompt,[subtasks])` data, pretraining script → `router.tlm`, `split()` heuristic, 4-bit I/O, UI + CLI training, interaction data collection
- [x] Inference usage: `Router.decompose()` autoregressively emits the decomposition with the trained weights (`decode_router_sequence`, `tiny-prose` fallback); `plan()` prefers it over heuristic `split()`
- [x] Context fix: blocks read only the first `emb` inputs, so concat-embedding left positions 1+ unread (proven: 0.0 output change) → mean-pooled window in `_embed_context`/`fast_logits` (shapes unchanged, old `router.tlm` files load)
- [ ] Same unread-context flaw exists in `SpecializedLM._embed_context` (`specialized.py:115`) → apply the same pooling there
- [x] Multi-subtask training data: `generate_router_training_data()` adds composite examples from consecutive sentences spanning distinct domains (`max_subtasks`, 1 = singles only)
- [x] Structural markers: `insert_markers()` tags `[EOT]` per sentence / `[EOL]` per line / `[EOS]` per section before training; single token IDs below the router block (`V-13..V-11`); training text renders the short forms yellow (`render_marked`, ANSI)
- [x] Markers enter training: `mark_and_encode()` is the canonical entry, wired into CLI `train` (yellow sample printed at start, `--no-markers` opt-out) and both UI training jobs (marked sample in the training log)
- [x] Training box shows markers: on train start the box displays the marked text (yellow tags in Tk, `MarkerHighlighter` in Qt, newlines preserved) and locks input; raw text is restored on completion/error/cancel (both toolkits)
- [x] De-hardwired runtime routing: `split()`/`route()`/`decompose()` and taskpad executors resolve models from the catalog only (unknown → `[]` / first catalogued model / dropped / clear error; only `router` stays literal)
- [ ] Same treatment for the rest: `router_data` slot map + `decode_router_sequence` + training-data keyword defaults; `ThinkEngine`/`HarnessRuntime` model defaults; `cmd_generate`/`cmd_train`/demo/benchmark `tinyLM` fallbacks; UI placeholders
- [ ] **MCP tool use**: tool-calling head, schema awareness, multi-step chains, result→context

### 2. Specialized Base (≤100K each)
- [x] 1-layer transformer, `create_model()` factory, 4-bit I/O, limit enforced at init
- [ ] **MCP tool use**: domain tool schemas within the 100K budget

### 3. Default Specialized Models (user-driven, unlimited count)
- [ ] Variable model count; auto `tiny-{domain}` names; seed configs (code/reason/prose/embed/sql/docs/shell/data); training priority queue; `tiny-code` (+SolidCraft/Git/LSP), `tiny-reason` (+Wolfram/proofs), `tiny-prose` (+search/docs/i18n), `tiny-embed` (CLS + contrastive, +vector DB/RAG), `tiny-sql` (+PG/MySQL/DuckDB), `tiny-docs`, `tiny-shell`, `tiny-data`; auto-create on domain gap

### 4. Multi-Model Runtime
- [x] `ModelRuntime` (lazy-load, shared tokenizer, `route`/`generate`, TTL + max-models budget)
- [ ] **MCP tool execution**: registry, sandbox, result streaming, timeouts/circuit breaker

### 5. Task System
- [x] Planner `plan()` (`scratchpad.py`): router `split()` → one `TaskPad` task per subtask (dynamic `model_to_task_type()` for any model count/naming, registry templates, `context_tokens` + empty `features` slot + `model_available` flag in `metadata`)
- [x] Cyclic path `run_cycle()` (`scratchpad.py`): plan once → execute pending → record each outcome in `Scratchpad`, until drained / no progress / `max_cycles`; returns `{cycles, tasks, done}`
- [ ] `Task` definition; DAG executor; built-ins (`generate/summarize/code/analyze/train/embed`); output types (`text/json/code/embedding/model`)
- [ ] **Feature extraction model** (`tiny-embed`): `extract_features()` filling `task.metadata["features"]` with the per-task prompt features

### 6. Collaborative Generation
- [ ] Shared `Scratchpad`; coordinator (token ranges, ordered merge); sync protocol; fallback reroute/replan

### 7. Training Pipeline
- [x] Generic `train()`, efficiency (pre-sampled negatives, grad accumulation, batched), data cleaning/sorting by domain, model selection with confidence, LR schedule, grad clipping, validation, early stopping, auto-adjust (plateau LR + clip relaxation), priority queue, cancel-with-save-if-improved
- [x] SGD with momentum in all training paths (`train()`, `_train_c()`, `train_parallel()`) + batch-averaged gradient updates (was divided by `batch_size` twice -> effective LR 16x too small) + C-kernel clip-after-averaging
- [x] Loss-spike rollback (`rollback_window`, config default 2, 0=off): snapshots every step, restores lowest-loss step in window on spike, skips bad update (`rollbacks` in result; `RollbackGuard` in `sched.py`; wired into `train()`/`_train_c()`/`train_parallel()` + Config/UI)
- [x] Multicore best-of-N trials (`train_multicore` in `tinylm/parallel.py`): one full training step per core from shared weights, shared probe batch picks the lowest-loss trial, winner's weights + velocity adopted; same result contract + `rollback_callback` (total divergence falls back to the best snapshot); fixed persistent-worker grad accumulation on the Python path (grads are now zeroed before each `backward()`)
- [x] Rollback monitor: `rollback_callback(step,loss,best)` in all training paths → training monitor logs each rollback and the Training tab shows the gap (current loss − smallest loss in run); completion/cancel summaries include the rollback count (Tk + Qt)
- [x] Auto-adjust in `train_parallel()` (plateau LR cuts + clip relaxation, same contract as `train()`) + Auto-adjust checkbox on the training page (both toolkits, default on from config)
- [x] Auto-adjust tunes LR, momentum (decayed toward 0 on plateaus to damp oscillation), and grad-clip in `train()`/`_train_c()`/`train_parallel()` — every adjusted value is stepped through `AutoAdjust`
- [x] Optimizers beyond SGD+momentum (`tinylm/sched.py`, wired into `train()` + `train_parallel()` parent update, selectable via `training.optimizer` config/UI): AdamW with bias correction (`adam`, best loss), Lion sign-momentum (`lion`, 1 state/param), Muon+AdamW hybrid (`muon`, Newton-Schulz orthogonalized momentum on ≤64-dim matrices + AdamW elsewhere — lowest loss measured). Shared `step()` contract (reads `p.grad`, in-place update, zeroes grads); per-family LR scales (`adam_lr_scale`, `lion_lr_scale`, `muon_scale`); unknown names raise instead of silently running SGD
- [x] C kernel: AdamW/Lion fused updates (`tl_train_step_opt` in `csrc/train_kernel.c`, `accel.train_step_opt`, `_train_c` dispatches sgd/adam/lion with C/Python parity at 1e-9; Muon/APOLLO stay Python-only, CUDA kernel stays SGD-only with dispatch guards so a wrong optimizer can never run silently)
- [x] `clean.py`: removes build artifacts (`*.so`, `__pycache__`, `*.pyc`) with `--dry-run`/`--checkpoints`/`--stats`/`--all`; never touches weights, merges, models/, trainingsData/, or history
- [x] Auto-adjust end-to-end in the UI path: `train_parallel()` honors `lr_scheduler`/`warmup_steps`/`min_lr`, frequency-weighted + adaptive negatives (per-window negs through workers), and reports live AA state (`aa_mult/aa_momentum/aa_grad_clip/aa_reduce/aa_bad/aa_best`) via `progress_cb` → Training tab panel (Qt); all `progress_cb` receivers accept `**kw`; `train_parallel` returns `kernel_used`
- [x] Population training (`train_population` in `tinylm/parallel.py`): N replicas train independently for `select_every` steps, lowest best-loss wins and its weights broadcast (elitist selection, optimizer state reset per generation); same result contract; keep `select_every` ≥ ~20 or it underperforms a single run
- [x] Batched compute + GPU workers: `train_parallel(batch_steps, use_gpu)` — gradient accumulation over K batches per update; workers route grads through the parallel CUDA kernel (`cuda_batch_loss_grads`, exact CPU parity) with per-task CPU fallback; fixed a pre-existing CUDA bug (all windows reused window 0's negatives); Training menu (Qt+Tk) exposes batch size / batch steps / GPU; `training.batch_steps` + `training.use_gpu` config keys
- [x] Tasks tab (Qt + Tk): TaskPad tasks + Scratchpad entries read from disk with detail view + Refresh
- [x] RWKV initial state: `RWKVModel` exported, full-tensor `get_tensors()` (save/load round-trips all blocks), `RWKVModel` registry factory (load-by-arch works), UI trains RWKV via `train()` (GPU WKV kernel when available, else CPU), `gpu_fast_logits` with CPU fallback
- [x] RWKV full GPU training (`csrc/rwkv_kernel.cu`, `accel_rwkv.py`): real WKV forward+backward kernel (one thread per window, BPTT with reverse-time state carry, both residuals, no-sqrt layernorm, simplified GELU; CPU parity 1e-14 grads / exact loss) + `rwkv_train_step` (averaged grads, clip-after-average matching `clip_grads` incl. `+1e-6`, SGD+momentum persistent velocity) + exact host-side `rwkv_generate_step`; `train()` routes RWKV+SGD to it via `use_c="cuda"` or auto (val/early-stop/checkpoint/cancel/non-SGD stay Python); fixed `RWKVModel.sampled_loss` to project the final hidden state through candidate rows (was silently truncating the full logits vector at `min(hidden, vocab)`)
- [ ] RWKV GPU in `train_parallel()` workers (`use_gpu` is still MLP-only there) + non-SGD optimizers on the RWKV kernel (Adam/Lion) + device-batched generation
- [ ] Same auto-adjust support in `train_multicore()`
- [x] C-path window sampling: `_train_c` chose on `list(starts)` (rebuilt an `len(ids)`-sized list per window, ~4.7s/1600 windows at 100k tokens) → chooses on the range directly (~7x end-to-end on the C path)
- [ ] Data loaders (`CodeDataset/ReasoningDataset/ProseDataset/EmbeddingDataset/SQLDataset/ShellDataset/DataAnalysisDataset/AutoDataset`)
- [ ] `train()` upfront negative presampling (`steps*batch_size*n_negatives` ints ≈ 8.2M/3s at defaults) → lazy chunked refill
- [ ] `train()` upfront negative presampling (`steps*batch_size*n_negatives` ints ≈ 8.2M/3s at defaults) → lazy chunked refill
- [ ] **Variable batch token router** (per-model trainers, no contamination)
- [ ] `gradient_accumulation` is accepted but ignored on the C kernel path (`_train_c` uses the fused per-batch step); make it effective or reject it there
- [ ] Router fine-tune (`train_router` on decomposition data)
- [ ] **User-driven training**: data → domain detect → create/update `tiny-{domain}` → queue

### 8. Tokenizer
- [x] `word_freq` cache
- [ ] Streaming `CorpusIterator`; cached incremental `extend()`; vocab growth triggers `sync_vocab` on all loaded models

### 9. Generation
- [x] Top-k / top-p / repetition penalty, unified `generate()`, speed opts, tokens/sec + live preview UI, `generate_with_preview()`, stream-callback early stop, `ModelRuntime` wiring, conversation-only chat, use-selected-model option, autostop (`stop_token_ids`/`stop_strings`/`min/max_tokens`)
- [x] Unbounded generation: `generate()` defaults to `max_tokens=None` and ends on structural markers (`stop_markers`, default `("EOS",)` ends after a finished section, resolved against the model vocab)
- [x] Same cap-free marker stop in `generate_with_preview()` (both), `ModelRuntime.generate`/`_generate_streaming` (shared `_resolve_stop_tokens`), and single-pass unbounded collaborative chunks
- [x] No mandatory cap in UI/config/CLI: `max_tokens` 0 = unlimited everywhere (`normalize_cap`), default 0, settings accept ≥ 0

### 10. Checkpointing
- [x] `Checkpoint`, save/load, auto-save callback, resume

### 11. Harness (automation engine: code/APIs/shell/MCP/workflows)
- [ ] Protection layer (allowlists, policy validation, audit log); MCP client (`mcp call/read`); `parse_tlm` → AST; `HarnessRuntime` + variables; commands (`prompt/task/assert/loop/branch/parallel/retry/save/load`); system commands (`code_run/shell/file_read/file_write/http_request/git/search`); assertions; `${var}` interpolation; parallel join; retry w/ backoff; `on_pass/on_fail/on_timeout` hooks; JSON/Markdown reporters; CLI `tinylm harness run`

### 11a. Skill System (local helper scripts, `skills/<name>/` + `skill.yaml`)
- [x] `opencodeSkill/readpy.py`: Python file-structure reader skill (`tree`/`treefull`/`class`/`function` + `classfull` alias) + `SKILL.md` agent index
- [x] Agent skill-use contract: `AGENTS.md` mandates loading the skill index first + `readpy tree` scouting before full reads (workflow step + Skills section)
- [ ] Manifest spec; auto-discovery + hot-reload; `run_skill()` sandbox (JSON IO, timeouts, no `Value` crossing); privilege levels + `skills_audit.jsonl`; CLI (`list/info/run/create`); UI section + arg forms + insert-into-chat; model-invoked skills; harness `skill run`; scaffolding

### 12. CLI / API
- [x] `tinylm` entry (main.py): router train, model mgmt, autotrain, `generate/train/eval/serve`, interactive prompt
- [x] `taskpad execute`/`run-all` actually execute via default generate-backed executors (task model → catalog → `./tinyLM` fallback; per-task `max_tokens`); graceful error when no model files are present
- [x] `python3 -m tinylm` demo guards `len(ids) <= model.ctx` (saved `tinyLM` ctx=2000 vs ~1802-token corpus): falls back to a fresh small-ctx model + `train()`/`_train_c()` raise `ValueError` instead of `IndexError`
- [ ] `train_parallel()`/`train_multicore()` still lack the `len(ids) <= ctx` `ValueError` guard (same `IndexError` via `random.choice` on empty `starts`)
- [x] Per-model training stats (`training_stats.json`): `train_with_stats()`/`apply_stats()` persist best/last loss + steps per model, snapshot best weights (`best_<name>.tlm`), and restore them on regression — loss only decreases; wired into CLI `train` and both UI training jobs (previous best shown at start, file saved only on improvement)
- [x] Warm start from best happens BEFORE vocab growth in every path (UI jobs, `train_with_stats`, router training); `load_best()` copies the shared prefix when the snapshot is smaller, so warm start survives vocab growth; `train_router` returns the result dict so router stats/warm-start work too
- [ ] Same stats wiring for router training (`train_router` paths)

### 13. Testing & Validation
- [x] Unit + integration tests (`test_all.py`); benchmarks (`benchmark.py`)
- [ ] Harness tests (`.tlm` scripts + assertions); CI via GitHub Actions (lint/test/bench)

### 14. Documentation (HIGH PRIORITY)
- [x] `README.md`, `CONFIG.md`
- [ ] `HARNESS.md`, `MODELS.md`, `API.md`

### 15. Build & Distribution (HIGH PRIORITY)
- [x] `main.py` entry (GUI/CLI/MCP), `build.sh`, C train kernel (~14x, pure-Python fallback)
- [ ] `main.py --mcp`; C tokenizer/quantize modules (`bpe_train`, quantize/dequantize); `setup.py`/`pyproject.toml`; `tinylm._c` extension

### 16. Code Structure (module splits)
- [x] `tinylm/`, `uitk/`, `uiqt/` packages; shims removed (`main.py` sole entry); low-token docs restructure (this task)
- [ ] **Split `test_all.py`**: one module per area (e.g. `tests/test_training.py`)

---

## Constraints (hard)

Same as AGENTS.md — stdlib only; never renumber IDs; `sync_vocab` on growth; main-thread UI only; multiprocessing under `__main__`; floats (never `Value`) to workers; in-place updates; 4-bit I/O; ≤100K specialized / 300K router; shared tokenizer; no fixed ctx; variable model count; no folder loader in UI.

---

## AI Scratchpad (internal)

Done: global `SCRATCHPAD`, `ThinkEngine` (`model.think`), TaskPad, scratchpad/taskpad CLIs.
