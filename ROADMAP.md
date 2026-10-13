# ScratchpadLM Task List

> Project: ScratchpadLM (public) · internal package `tinylm` / `TinyLM`.

Done to date: fused `matvec_dot`, persistent `train_parallel` workers, O(1)-depth loss accumulation (~2x sequential / ~3x parallel); `tinylm/` + `uitk/` + `uiqt/` module splits with shims removed.

## Backlog (ordered, atomic)

### 0. Foundations
- [x] Config, registries, task registry, tokenizer singleton, 4-bit save/load, param counter, Config wired into UI

### 0a. Control UI (HIGH PRIORITY)
- [x] Prompt interface, tabbed layout (Chat/Training/Model Info/Admin/Settings/Logs), PySide6 primary + Tk fallback (`ui_base.py`, auto-detect + `SCRATCHPADLM_UI` override), chat layout (tabs/sidebar/bubbles/input/send/thumbs→RatingCollector→AutoTrainer), theme system (dark/light/system, live switch), auto model naming, train-vs-create advisor, model selector, training monitor, generation controls, tokenizer manager, config editor, log viewer, model inspector, layout polish
- [ ] **UI Toolkit Migration** (parent)
- [ ] **Harness Tab (PySide6)**: privilege selector, root path + browse, mode toggle, privilege help, path validation, status indicator
- [ ] **Image processor**: upload/preview, resize/normalize, vision-model prompts
- [ ] **Training data provider**: file/text/URL/folder-watch input, preview, domain detection, train/val split
- [ ] **Variable batch tokenizer**: mixed-domain batches routed per model, no cross-domain contamination
- [ ] **Control panel**: PySide6 dashboard (Tk fallback)

### 0b. Dynamic Model Management
- [x] `ModelSpec`, `ModelCatalog` (JSON), `ModelLoader`, `ModelCreator`, `ModelUpdater`, catalog CLI, auto-discovery, router↔catalog routing
- [ ] **Harness integration**: `model load/create` in `.tlm` scripts

### 0c. Auto-Training
- [x] `RatingCollector` (thresholds 4/5 high, 2/5 low), `ModelSelector`, `AutoTrainer` (batch ≥8, 60s wait, few-shot fine-tune, `autotrain.log`), low-rating handler, dedup, rate limits, per-model metrics
- [ ] **CLI**: `autotrain status/enable/disable/config`
- [ ] **Harness integration**: `autotrain on/off/config`, `rate last <1-5>`

### 1. Router (300K, pretrained, frozen)
- [x] `RouterConfig`, 2-layer transformer, forward + fast inference, sampled-loss training, `(prompt,[subtasks])` data, pretraining script → `router.tlm`, `split()` heuristic, 4-bit I/O, UI + CLI training, interaction data collection
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
- [ ] `Task` definition; planner (`plan()` via router+registry); DAG executor; built-ins (`generate/summarize/code/analyze/train/embed`); output types (`text/json/code/embedding/model`)

### 6. Collaborative Generation
- [ ] Shared `Scratchpad`; coordinator (token ranges, ordered merge); sync protocol; fallback reroute/replan

### 7. Training Pipeline
- [x] Generic `train()`, efficiency (pre-sampled negatives, grad accumulation, batched), data cleaning/sorting by domain, model selection with confidence, LR schedule, grad clipping, validation, early stopping, auto-adjust (plateau LR + clip relaxation), priority queue, cancel-with-save-if-improved
- [x] SGD with momentum in all training paths (`train()`, `_train_c()`, `train_parallel()`) + batch-averaged gradient updates (was divided by `batch_size` twice -> effective LR 16x too small) + C-kernel clip-after-averaging
- [x] Loss-spike rollback (`rollback_window`, config default 2, 0=off): snapshots every step, restores lowest-loss step in window on spike, skips bad update (`rollbacks` in result; `RollbackGuard` in `sched.py`; wired into `train()`/`_train_c()`/`train_parallel()` + Config/UI)
- [x] Multicore best-of-N trials (`train_multicore` in `tinylm/parallel.py`): one full training step per core from shared weights, shared probe batch picks the lowest-loss trial, winner's weights + velocity adopted; same result contract + `rollback_callback` (total divergence falls back to the best snapshot); fixed persistent-worker grad accumulation on the Python path (grads are now zeroed before each `backward()`)
- [x] Rollback monitor: `rollback_callback(step,loss,best)` in all training paths → training monitor logs each rollback and the Training tab shows the gap (current loss − smallest loss in run); completion/cancel summaries include the rollback count (Tk + Qt)
- [ ] Data loaders (`CodeDataset/ReasoningDataset/ProseDataset/EmbeddingDataset/SQLDataset/ShellDataset/DataAnalysisDataset/AutoDataset`)
- [ ] **Variable batch token router** (per-model trainers, no contamination)
- [ ] `gradient_accumulation` is accepted but ignored on the C kernel path (`_train_c` uses the fused per-batch step); make it effective or reject it there
- [ ] Router fine-tune (`train_router` on decomposition data)
- [ ] **User-driven training**: data → domain detect → create/update `tiny-{domain}` → queue

### 8. Tokenizer
- [x] `word_freq` cache
- [ ] Streaming `CorpusIterator`; cached incremental `extend()`; vocab growth triggers `sync_vocab` on all loaded models

### 9. Generation
- [x] Top-k / top-p / repetition penalty, unified `generate()`, speed opts, tokens/sec + live preview UI, `generate_with_preview()`, stream-callback early stop, `ModelRuntime` wiring, conversation-only chat, use-selected-model option, autostop (`stop_token_ids`/`stop_strings`/`min/max_tokens`)

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
- [x] `python3 -m tinylm` demo guards `len(ids) <= model.ctx` (saved `tinyLM` ctx=2000 vs ~1802-token corpus): falls back to a fresh small-ctx model + `train()`/`_train_c()` raise `ValueError` instead of `IndexError`
- [ ] `train_parallel()`/`train_multicore()` still lack the `len(ids) <= ctx` `ValueError` guard (same `IndexError` via `random.choice` on empty `starts`)

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
