# ScratchpadLM Task List

> **Project name**: ScratchpadLM (public-facing)  
> **Internal package**: `tinylm` / `TinyLM` (unchanged)

## Done
- [x] Fused `matvec_dot` — single autograd node per layer
- [x] Persistent workers in `train_parallel` — no model rebuild per step
- [x] List-based loss accumulation — O(1) graph depth
- [x] ~2x sequential, ~3x parallel speedup

---

## Backlog (ordered, atomic steps)

### 0. Foundations (prerequisites)
- [x] **Config system**: `Config` class (YAML/JSON → dict, validation, defaults)
- [x] **Model registry**: `ModelRegistry` — `register(name, factory)`, `get(name)`, `list()`
- [x] **Task registry**: `TaskRegistry` — `register(name, model_name, template)`, `get(name)`
- [x] **Shared tokenizer singleton**: `TokenizerManager` — single `BPETokenizer` instance, `extend()`, `save()`, `load()`
- [x] **4-bit quantized save/load per model**: `save_model(model, path)`, `load_model(path)` — works for any architecture
- [x] **Parameter counter**: `count_params(model)` — enforces ≤100K / 300K limits at init
- [x] **Wire Config into ui.py**: Replace hardcoded `VOCAB_SIZE, CTX, EMB, HIDDEN` with `Config`

### 0a. Control UI (HIGH PRIORITY)
- [x] **Prompt interface**: Text input box for prompting the AI, send button, conversation history
- [x] **Tabbed UI layout**: Reorganize UI into tabs — **Chat** (prompt + history), **Training** (data input + controls + monitor), **Model Info** (inspector + tensor details), **Admin** (model list, create/delete, catalog mgmt), **Settings** (generation params, training config, tokenizer config)
- [ ] **UI Toolkit Migration**: Migrate from Tkinter to **PySide6 (Qt6)** as primary UI toolkit, with Tkinter as fallback
  - [ ] Abstract UI layer with common interface for both toolkits
  - [ ] Implement PySide6 version of all tabs (Chat, Training, Model Info, Admin, Settings, Logs)
  - [ ] Keep Tkinter implementation as fallback for environments without Qt
  - [ ] Auto-detect available toolkit at startup (PySide6 preferred, Tkinter fallback)
  - [ ] Consistent styling/theme across both toolkits
- [ ] **Chat Layout (PySide6)**: Modern chat interface layout
  - [ ] **Tabs at top**: Main navigation tabs (Chat, Training, Model Info, Admin, Settings, Logs, **Harness**)
  - [ ] **Chats list at left**: Sidebar with conversation list/sessions
  - [ ] **Chat area center**: Main chat display with messages
  - [ ] **Chat input at bottom**: Text input area at bottom of chat
  - [ ] **Send button right of chat input**: Send button positioned to the right of input field
  - [ ] **Thumbs up/down on each answer**: Every AI response has thumbs up/down buttons for reinforcement learning feedback
  - [ ] **Reinforcement Learning integration**: Thumbs up/down feeds into auto-training system (RatingCollector → AutoTrainer)
- [ ] **Harness Tab (PySide6)**: Harness configuration and control
  - [ ] **Privilege Level Selector**: Dropdown/combo box to select AI privilege level (Read-only, Read-Write, Full Access, Admin)
  - [ ] **Harness Root Path**: Text input with browse button to set harness root directory path
  - [ ] **Harness Mode Toggle**: Slider/toggle switch to enable/disable harness mode
  - [ ] **Privilege Level Descriptions**: Tooltip/help text explaining each privilege level
  - [ ] **Path Validation**: Real-time validation of harness root path existence/permissions
  - [ ] **Harness Status Indicator**: Visual indicator showing current harness mode (enabled/disabled) and active privilege level
- [ ] **Image processor**: Image upload/preview, preprocessing (resize, normalize), vision model integration for multimodal prompts
- [ ] **Training data provider**: User provides training data directly — file upload, text paste, URL fetch, folder watch, data preview, auto domain detection, train/val split
- [ ] **Auto model naming**: System generates model name from training data domain — shows preview `tiny-{detected_domain}` before creation
- [ ] **Train vs Create advisor**: Analyzes user data → "Fine-tune existing tiny-X (92% match)" vs "Create new tiny-Y (new domain)" with confidence
- [ ] **Variable batch tokenizer**: Training batches with mixed-domain detection — routes tokens to appropriate model trainers, prevents single-model training on heterogeneous data
- [ ] **Control panel**: PySide6-based dashboard with model status, training controls, generation settings (Tkinter fallback)
- [x] **Model selector**: dropdown to switch between registered models (router + specialized)
- [x] **Training monitor**: progress bar with step counter + live loss (loss chart/LR display still open)
- [x] **Generation controls**: temperature, top-k, top-p, repetition penalty sliders
- [x] **Tokenizer manager**: view vocab size, learned patterns, extend vocab button
- [ ] **Config editor**: YAML editor with validation, save/load presets
- [x] **Log viewer**: scrollable training/generation logs with filtering
- [x] **Model inspector**: parameter count, memory usage, architecture summary
- [x] **UI polish & layout fixes**: Fix spacing issues, elements going off-screen, consistent padding/margins, responsive resizing, scrollable tabs for overflow content, improve visual hierarchy and alignment

### 0b. Dynamic Model Management (runtime)
- [x] **ModelSpec**: dataclass `{name, path, domain_keywords, param_count, config, created_at, metrics}`
- [x] **ModelCatalog**: persistent registry (JSON file) — `add(spec)`, `remove(name)`, `get(name)`, `list()`, `find_by_domain(keywords)`
- [x] **ModelLoader**: `load_model(name)` — loads from catalog path, injects shared tokenizer, validates param limit
- [x] **ModelCreator**: `create_model(name, domain_keywords, base_config=None)` — instantiates new specialized model, saves to catalog, returns handle
- [x] **ModelUpdater**: `update_model(name, new_config)` — hot-swaps config, preserves weights where shapes match
- [x] **Catalog CLI**: `tinylm model list`, `tinylm model add <name> --domain "code,python"`, `tinylm model remove <name>`, `tinylm model info <name>`
- [x] **Auto-discovery**: on startup, scan `models/` dir, register any `.tlm` files not in catalog
- [x] **Router integration**: router reads catalog `domain_keywords` for routing decisions
- [ ] **Harness integration**: `model load <name>`, `model create <name> --domain "..."` commands in `.tlm` scripts

### 0c. Auto-Training System (online learning from feedback)
- [x] **RatingCollector**: `collect(prompt, response, model_name, rating, metadata)` — stores `(prompt, response, model, rating)` in `feedback.jsonl`
- [x] **Rating thresholds**: `HIGH_THRESHOLD=4/5`, `LOW_THRESHOLD=2/5` — configurable
- [x] **ModelSelector**: `select_model_for_input(prompt)` — uses router + catalog domain_keywords → best model
- [x] **AutoTrainer**: background worker that:
  - Monitors `feedback.jsonl` for new high-rated entries (rating ≥ HIGH_THRESHOLD)
  - Groups by model_name
  - Batches entries (min batch=8, max_wait=60s)
  - Calls `train(model, batch_data, steps=10, lr=0.01)` — few-shot fine-tune
  - Updates model in catalog, saves checkpoint
  - Logs: `autotrain.log` — timestamp, model, batch_size, loss_before, loss_after
- [x] **Low-rating handler**: rating ≤ LOW_THRESHOLD → flags for review, optionally trains "avoidance" (negative examples)
- [x] **Deduplication**: hash prompts, skip near-duplicates (edit distance < 0.2)
- [x] **Rate limiting**: max 1 auto-train per model per 5 min, max 100 steps/day per model
- [ ] **CLI**: `tinylm autotrain status`, `tinylm autotrain enable/disable`, `tinylm autotrain config --threshold 4 --batch 16`
- [ ] **Harness integration**: `autotrain on`, `autotrain off`, `autotrain config`, `rate last <1-5>` commands
- [x] **Metrics**: tracks per-model `avg_rating`, `train_count`, `last_trained`, `improvement_delta`

### 1. Router Model (300K params, pretrained)
- [x] Router config: `RouterConfig(vocab, ctx, emb, hidden, n_layers=2)`
- [x] Router architecture: 2-layer transformer (attention + FFN, residuals, LayerNorm)
- [x] Router forward: input tokens → logits (with fast inference)
- [x] Router training: sampled_loss compatible with train()
- [x] Router training data format: `(prompt, [subtasks])` pairs
- [x] Router pretraining script: `train_router(corpus, steps=10000)` → saves `router.tlm`
- [x] Router inference: `router.split(prompt)` → list of subtasks with model assignments (heuristic)
- [x] Router quantization: 4-bit save/load compatible
- [x] **Router integration in UI**: Router used for every user prompt, routes to best model
- [x] **Router training in UI**: `tinylm router train` CLI and UI button for training on collected data
- [x] **Router training data collection**: Automatic collection from user interactions
- [x] **Router pretraining**: CLI `tinylm router train` pretrains router on corpus
- [ ] **MCP tool use**: Router can invoke MCP tools during inference — tool calling head, tool schema awareness, multi-step tool chains, result integration into context

### 2. Specialized Model Base (≤100K params each)
- [x] Base config: `ModelConfig(vocab, ctx, emb, hidden, n_layers=1, model_type)`
- [x] Base architecture: 1-layer transformer block (attention + FFN, residuals, LayerNorm)
- [x] Model factory: `create_model(config)` — returns model instance
- [x] Model save/load: uses shared 4-bit format
- [x] Param limit enforcement: raises if `count_params > 100_000`
- [ ] **MCP tool use**: Specialized models can invoke MCP tools — tool calling head, domain-specific tool schemas (e.g., SolidCraft for tiny-code), result integration, stays within 100K param budget

### 3. Default Specialized Models (variable count, user-driven)
- [ ] **Variable model count**: No fixed limit — user provides training data for any number of domains, models created on demand
- [ ] **Auto-generated model names**: System generates names from training data — domain analysis → `tiny-{domain}` (e.g., `tiny-rust`, `tiny-k8s`, `tiny-biology`)
- [ ] **Seed models**: Pre-defined starter configs for common domains (code, reason, prose, embed, sql, docs, shell, data) — user can extend
- [ ] **Model prioritization**: Training queue with priority scoring — domain coverage gaps, performance metrics, data availability → decides which models to train first
- [ ] `tiny-code`: config optimized for code (larger hidden, code-specific pretraining) — MCP tools: SolidCraft (CAD sketches), Git (version control), LSP (code analysis)
- [ ] `tiny-reason`: config optimized for logic (deeper, math/logic pretraining) — MCP tools: WolframAlpha (computation), Proof assistants (Lean, Coq)
- [ ] `tiny-prose`: config optimized for text (standard) — MCP tools: Web search, Document APIs, Translation
- [ ] `tiny-embed`: no LM head, outputs `[CLS]` embedding, contrastive loss — MCP tools: Vector DB (search/index), RAG pipelines
- [ ] `tiny-sql`: SQL/databases — MCP tools: PostgreSQL, MySQL, DuckDB
- [ ] `tiny-docs`: documentation/manuals — MCP tools: Confluence, Notion, GitBook
- [ ] `tiny-shell`: shell/CLI commands — MCP tools: terminal, kubectl, AWS CLI
- [ ] `tiny-data`: data analysis/visualization — MCP tools: Pandas, Plotly, Jupyter
- [ ] **Dynamic model creation**: Auto-create new specialized models when domain gap detected — analyzes user-provided training data, generates config, registers in catalog

### 4. Multi-Model Runtime
- [x] `ModelRuntime` class: loads router + all specialized models at startup (lazy-load)
- [x] Shared tokenizer injected into all models
- [x] `runtime.route(prompt)` → router splits → dispatches to specialized models
- [x] `runtime.generate(prompt)` → collaborative generation coordinator
- [x] Memory budget: lazy-load models, unload unused after timeout (TTL + max models)
- [ ] **MCP tool execution**: Runtime executes MCP tool calls from router — tool registry, invocation sandbox, result streaming back to model context, timeout/circuit breaker

### 5. Task System
- [ ] Task definition: `Task(name, model_name, prompt_template, output_parser)`
- [ ] Planner: `plan(prompt)` → uses router + task registry → `[TaskInstance]`
- [ ] Executor: runs task graph (DAG), handles dependencies, passes outputs
- [ ] Built-in tasks: `generate`, `summarize`, `code`, `analyze`, `train`, `embed`
- [ ] Task output types: `text`, `json`, `code`, `embedding`, `model`

### 6. Collaborative Generation
- [ ] Shared scratchpad: `Scratchpad` — append-only, `{step, model, tokens, metadata}`
- [ ] Coordinator: assigns token ranges / sections to models, merges in order
- [ ] Sync protocol: models write to scratchpad, coordinator emits final stream
- [ ] Fallback: if model fails, reroute to another or use router to replan

### 7. Training Pipeline
- [x] `train(model, data, steps, config)` — works for any model type
- [x] **Training efficiency improvements**: pre-computed negative samples, gradient accumulation, optimized sampled softmax, batched processing, reduced memory allocations
- [ ] **Data cleaning & sorting**: normalize text (whitespace, unicode, encoding fixes), deduplicate lines, strip boilerplate, sort/classify cleaned data by domain → feeds per-model training queues
- [ ] Data loaders: `CodeDataset`, `ReasoningDataset`, `ProseDataset`, `EmbeddingDataset`, `SQLDataset`, `ShellDataset`, `DataAnalysisDataset`, `AutoDataset` (from user data)
- [ ] **Variable batch token router**: Analyzes batch tokens → routes to correct model trainer (code→tiny-code, math→tiny-reason, etc.), prevents cross-domain contamination
- [ ] **Model selection for training**: Given user-provided training data, evaluates catalog → returns (existing_model_to_finetune OR new_model_spec_with_autogenerated_name) with confidence score
- [x] LR scheduler: `CosineWithWarmup(lr, warmup_steps, total_steps)`
- [x] Gradient clipping: `clip_grads(params, max_norm=1.0)`
- [x] Validation: `evaluate(model, val_data)` → perplexity / accuracy
- [x] Early stopping: patience, min_delta, restore best
- [x] **Automatic adjustment**: plateau LR reduction (`AutoAdjust`) + auto grad-clip relaxation — wired into `train()`, `_train_c()`, C kernel path
- [ ] Router fine-tune: `train_router(router, decomposition_data, steps)`
- [ ] **Training priority queue**: Scores models by (domain_gap * data_volume * performance_delta) → trains highest priority first
- [ ] **User-driven training**: UI accepts training data → auto-detects domain → creates/updates model with generated name → queues for training

### 8. Tokenizer Enhancements
- [ ] `word_freq` cache: `save_word_freq(path)`, `load_word_freq(path)`
- [ ] Streaming corpus: `CorpusIterator(folder, pattern)` — yields chunks
- [ ] `extend()` uses cache, merges incrementally
- [ ] Vocab growth triggers `sync_vocab()` on all loaded models

### 9. Generation Improvements
- [x] Top-k sampling: `sample_top_k(logits, k)` (implemented in `_sample_next_token`)
- [x] Top-p (nucleus) sampling: `sample_top_p(logits, p)` (implemented in `_sample_next_token`)
- [x] Repetition penalty: `apply_repetition_penalty(logits, generated, penalty)` (implemented in `_sample_next_token`)
- [x] Unified `generate(model, prompt, config)` with all options
- [x] **Generation speed optimizations**: optimized `_sample_next_token` with cached softmax, efficient sorting
- [x] **UI tokens/sec display**: live tokens/sec in status bar during generation
- [x] **Live token preview**: streaming token output with "Generate with Preview" button showing tokens/sec
- [x] `generate_with_preview()`: detailed timing stats and console output

### 10. Checkpointing
- [x] `Checkpoint` class: `model_state`, `optimizer_state`, `step`, `config`
- [x] `save_checkpoint(path, checkpoint)`, `load_checkpoint(path)`
- [x] Auto-save callback: `every_n_steps(n, callback)` (via `checkpoint_every` parameter)
- [x] Resume: `train(model, data, steps, resume_from=path)` (via `resume_from_checkpoint`)

### 11. Harness (Automation Engine) — **System Interaction Layer**
> The Harness is the **automation engine** that lets models interact with external systems: running code, calling APIs, executing shell commands, using MCP tools, and orchestrating complex workflows. It's the bridge between model reasoning and real-world actions (coding, file ops, web search, etc.).

- [ ] **Protection layer**: Anti-circumvention guard — enforces command restrictions, prevents AI from bypassing allowlists/blocklists, validates all tool calls against policy, audit logging of all executions
- [ ] **MCP support**: Model Context Protocol client — connect to MCP servers, discover tools/resources, call tools, read resources, use in `.tlm` scripts via `mcp call` / `mcp read` commands
- [ ] **Parser**: `parse_tlm(script)` → AST (commands, assertions, control flow)
- [ ] **Runtime**: `HarnessRuntime` — executes AST, manages variables, models
- [ ] **Commands**: `prompt`, `task`, `assert`, `loop`, `branch`, `parallel`, `retry`, `save`, `load`
- [ ] **System interaction commands**: `code_run`, `shell`, `file_read`, `file_write`, `http_request`, `git`, `search`
- [ ] **Assertions**: `contains`, `matches`, `json_valid`, `python_runs`, `loss_lt`, `latency_lt`
- [ ] **Variables**: `$var = capture(command)`, `${var}` interpolation
- [ ] **Parallel**: `parallel [cmd1, cmd2]` → concurrent execution, join results
- [ ] **Retry**: `retry N { cmd }` — exponential backoff, max attempts
- [ ] **Hooks**: `on_pass`, `on_fail`, `on_timeout` — user callbacks
- [ ] **Reporter**: `JSONReporter`, `MarkdownReporter` — summary + details
- [ ] **CLI**: `tinylm harness run script.tlm --reporter json --output report.json`

### 12. CLI / API
- [x] `tinylm` entry point (argparse subcommands) - basic implementation in main.py
- [x] `tinylm router train` - pretrain router on corpus
- [x] `tinylm model list/info/create` - model management
- [x] `tinylm autotrain status/enable/disable/config` - auto-training control
- [x] `tinylm autotrain low-rated/review/avoidance` - low-rating handler
- [x] `tinylm generate` - generate text from prompt
- [x] `tinylm train` - train model on data
- [ ] `tinylm eval --checkpoint path --data val.jsonl`
- [ ] `tinylm serve --port 8080` — HTTP `/generate`, `/embed`, `/route`
- [ ] `tinylm "prompt"` — interactive, auto-plans via router + tasks

### 13. Testing & Validation
- [x] Unit tests: autograd, tokenizer, model forward/backward, quantize/dequantize (`test_all.py`)
- [x] Integration tests: train → save → load → generate (`test_all.py`)
- [ ] Harness tests: run `.tlm` scripts, verify assertions
- [x] Benchmark script: `bench_train()`, `bench_generate()`, `bench_router()` (in `benchmark.py`)
- [ ] CI: GitHub Actions (lint, test, bench)

### 14. Documentation (HIGH PRIORITY)
- [ ] `README.md` — quickstart, architecture, constraints
- [ ] `CONFIG.md` — all config options
- [ ] `HARNESS.md` — script DSL reference, examples
- [ ] `MODELS.md` — adding custom specialized models
- [ ] `API.md` — Python API, HTTP API

### 15. Build & Distribution (HIGH PRIORITY)
- [x] `main.py` — entry point: no args → GUI boot, `--cli` → CLI, `--mcp` → MCP system
- [x] `main.py --cli` — CLI implemented (router train, model mgmt, autotrain, generate, train)
- [ ] `main.py --mcp` — implement MCP system → section 11 (MCP support)
- [x] `build.sh` — build script for optional C extensions
- [x] C source for training hot path: `csrc/train_kernel.c` + `accel.py` (ctypes) — fused sampled-softmax fwd/bwd + SGD with gradient clipping (`max_norm`), ~14x faster, pure-Python fallback
- [ ] C source for remaining modules (tokenizer BPE, quantize/dequantize)
- [ ] `setup.py` / `pyproject.toml` for pip installable package
- [ ] Optional C extension: `tinylm._c` with `bpe_train`, `quantize`, `dequantize`
- [x] Fallback to pure Python when C not available (training; enforced by `accel.AVAILABLE` check)

---

## Constraints (hard)
- [ ] Zero external dependencies (stdlib only)
- [ ] No folder corpus loader in UI
- [ ] Never renumber token IDs
- [ ] Never recreate model on vocab growth (use `sync_vocab`)
- [ ] Tkinter calls only on main thread
- [ ] Never send `Value` objects to subprocesses
- [ ] In-place updates only (`p.data -= ...`)
- [ ] All models 4-bit quantized (save/load via `save_model`/`load_model`)
- [ ] Max 100K parameters per specialized model (enforced at init)
- [ ] Router model: 300K params, pretrained, frozen by default
- [ ] Single shared tokenizer — all models use same vocab
- [ ] No fixed context length — dynamic positional embeddings
- [ ] Variable model count — register/unregister at runtime

---

## AI Scratchpad (internal)
- [x] `scratchpad = []` — list of `{step, thought, result, model}`
- [x] `model.think(prompt, steps=3)` → final answer + trace (ThinkEngine)
- [x] Used by `code`, `analyze`, `reason` tasks (TaskPad with TaskType)
- [x] Shared across models during collaborative generation (global SCRATCHPAD)
- [x] **Scratchpad CLI**: `tinylm scratchpad show/clear/add`
- [x] **TaskPad**: task management with dependencies, status tracking
- [x] **TaskPad CLI**: `tinylm taskpad list/create/execute/run-all`