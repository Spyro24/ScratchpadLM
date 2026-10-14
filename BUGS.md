# BUGS.md — known bugs in ScratchpadLM

> Legend: **[exec]** = reproduced by running code, **[read]** = verified by
> reading the cited lines. Severity: HIGH = crash / data-loss / silent-wrong
> training or security hole; MEDIUM = wrong behavior in a reachable path;
> LOW = cosmetic / latent / narrow. Line numbers refer to the current tree.

## Config (`config.py`, `config.yaml`) — [exec]+[read]

- **HIGH [exec]: missing sections alias the class-level `DEFAULTS` dict.**
  `Config._merge_defaults` (`config.py:235-247`) assigns
  `result[key] = default_val` by reference. Mutating one instance poisons all
  future ones: `c1._data['model']['vocab_size']=999` → `Config.from_dict({})`
  then reports `999`. Worse, `to_dict()` (`config.py:293-295`) is a shallow
  copy, so `d=cfg.to_dict(); d['training']['lr']=-99` rewrites the live
  config *and* the class defaults (observed: all later `Config` creations in
  the process failed validation with `got int`).
- **HIGH [exec]: block lists crash the YAML parser.**
  `load_yaml` (`config.py:13-70`) pushes only `{}` for empty values
  (`:62-66`), never a list, so valid YAML like
  `stop_token_ids:\n  - 1` raises `ConfigError: list item outside of list`.
  Only inline `[]` parses.
- **MEDIUM [exec]: non-empty inline lists misparse as strings.**
  `_parse_value` (`config.py:73-100`) special-cases `'[]'` only, so
  `x: [1, 2]` yields the string `'[1, 2]'`, which then fails list validation
  with a confusing message.
- **MEDIUM [exec]: unknown keys are dropped at top level but kept (unvalidated) nested.**
  `_merge_defaults` iterates `DEFAULTS` keys only, so `{'extra': ...}` is
  silently dropped, while `{'training': {'bogus_key': 1}}` is kept with no
  error. Typos/renames fail silently in both directions.
- **MEDIUM [exec]: `workers`/`n_negatives_start` skip type checks.**
  The `expected_type is not type(None)` guard (`config.py:268`) skips
  `isinstance` for `None`-typed fields, so `workers: 2.5` validates and flows
  into pool sizing, while `workers: 'foo'` raises raw `TypeError` instead of
  `ConfigError`.
- **LOW [exec]: float fields reject YAML ints; int fields accept bools.**
  `lr: 1` → `ConfigError: expected float, got int` (`config.py:177`); conversely
  `isinstance(True, int)` lets `steps: True` through. Hostile to hand-written configs.
- **LOW [exec]: non-dict section raises the wrong exception.**
  `Config.from_dict({'model': 5})` → `TypeError` (`config.py:258-266`,
  `section_data` iteration + `get()` at `:297-299` assume dicts) instead of `ConfigError`.
- **LOW [read]: stale `n_tokens` key in shipped config.**
  `config.yaml:41` sets `n_tokens: 60` next to `max_tokens: 0` (= unlimited).
  No UI/CLI path reads `generation.n_tokens`, and the schema has no rule for
  it, so it is inert yet reads as a 60-token cap (`uitk/act_chat.py:45-77`,
  `uiqt/chat.py:191-234` use `max_tokens`; `generate()` keeps `n_tokens` only
  as a deprecated kwarg at `tinylm/generation.py:176-178`).

## Tokenizer / registry / model management — [exec]+[read]

- **HIGH [exec]: `BPETokenizer.train()` silently renumbers existing IDs.**
  `tokenizer.py:93-120` never clears `merges`/`vocab` and restarts
  `next_id = 256`, so a second `train()` overwrites IDs from 256 up.
  Observed: `vocab[256]` changed `b'he'` → `b'er'`, old encodings decode to
  garbage (`'hello world'` → `'te h'`). Violates the never-renumber rule;
  use `extend()`. (Startup/re-train paths that call `train()` twice corrupt
  saved models.)
- **MEDIUM [read]: `TOKENIZER_MANAGER.load()` desyncs the growth target.**
  `registry.py:158-171`: `load()` never updates `_vocab_size` (default
  `1000` at `:141`), while no-arg `extend()` targets it and the `vocab_size`
  property reports actual length. After loading a larger file, no-arg
  `extend()` is a silent no-op (target < actual); after a smaller file it
  over-grows. Also falsy-but-explicit `0` falls back to defaults
  (`vocab_size or self._vocab_size`, `min_frequency or ...` at `:168-169`),
  so `extend(text, min_frequency=0)` silently uses `2`. (LOW for the `or` part.)
- **LOW [read]: `vocab`/`merges` properties expose live dicts.**
  `registry.py:183-189` return the singleton's internal dicts by reference;
  any caller can corrupt shared tokenizer state (`m.vocab[9999] = b'X'` sticks).
- **MEDIUM [exec]: corrupt catalog JSON crashes load; writes are non-atomic.**
  `model_mgmt.py:61-71`: `'{corrupt'` → raw `json.JSONDecodeError` (no
  recoverable error), and `_save` truncates-then-writes, so a mid-write crash
  or concurrent writers leave a torn catalog that then fails to load.
- **MEDIUM [exec]: one unknown field breaks the whole catalog.**
  `ModelSpec.from_dict` (`model_mgmt.py:43-45`) does `cls(**data)`, so a
  hand-added/forward-compat field raises `TypeError` on catalog load.
- **LOW [read]: `catalog.update()` silently drops typos, returns `True`.**
  `model_mgmt.py:104-113` skips unknown fields via `hasattr` yet reports success.
- **LOW [read]: unsanitized model name in `create_model`.**
  `model_mgmt.py:303` (`models_dir / f"{name}.tlm"`) allows path traversal
  (`../../evil`); local-only vector. Also `config.setdefault("vocab_size",
  ...)` (`model_mgmt.py:279-285`) keeps a caller-supplied mismatched vocab
  instead of enforcing equality with the shared tokenizer.

## Transformer / specialized models — [exec]

- **HIGH [exec]: `SpecializedLM` ignores all context past the first token.**
  `_embed_context` returns `ctx*emb` values (`tinylm/specialized.py:115-127`)
  into blocks built with `d_model=emb` (`:80`); truncation keeps only the
  first `emb` entries. Observed: varying tokens 1–3 leaves logits
  bit-identical, varying token 0 changes them.
- **HIGH [exec]: `attention()` soft-maxes globally, `fast_attention()` per row.**
  `tinylm/transformer.py:53-92` vs `:95-110` disagree on identical inputs, so
  `TransformerBlock.forward` and `fast_forward` (hence training vs inference)
  diverge.
- **HIGH [exec+read]: `attention()` forward/backward are both wrong.**
  Output `j` is `v[j]` scaled by a column-sum of probs — never a mix of values
  (`:79-90`, indices transposed vs the documented row-wise mix); backward
  spreads `out[j].grad` across every `v[i]` (`:82-86`); and `score` nodes get
  no `_back` with probs computed from detached `.data`, so `q`/`k` grads are
  always `0.0` — the Q/K half of `W_qkv` never learns (observed).
- **HIGH [exec]: `layer_norm` backward drops mean/variance terms.**
  `tinylm/transformer.py:23-34` back-props `w*inv_std` only; finite-difference
  check disagrees (e.g. autograd `[-1.58, 0.81, -0.69]` vs numeric
  `[-1.67, -0.51, 2.18]`). Every `TransformerBlock`/`SpecializedLM` trains on
  wrong gradients. (Contrast the no-sqrt RWKV layernorm, whose backward was
  verified exact.)
- **MEDIUM [exec]: `TransformerBlock` cannot be saved standalone.**
  It has `get_tensors` (`tinylm/transformer.py:181`) but no `get_config`,
  so `save_model(TransformerBlock(8,16), ...)` raises `AttributeError`.
- **MEDIUM [exec]: `attention` silently truncates long `k`/`v`, crashes on short.**
  Everything is sized from `len(q)` (`tinylm/transformer.py:56,80,95`):
  `len(q)=1, len(k)=3` returns 1 output using only first elements; shorter
  inputs raise bare `IndexError`.
- **LOW [exec]: `layer_norm([],[],[])` raises bare `ZeroDivisionError`**
  (`tinylm/transformer.py:12,44`, `sum(...)/n` with `n = 0`).
- **LOW [read]: `SpecializedConfig.hidden` is a dead knob.**
  Declared (`tinylm/specialized.py:17`), serialized (`:28`), set by presets
  (`:187`), never read by `SpecializedLM` (arch driven by `emb`); the param
  estimate (`:49`) also miscounts FFN bias (`2*d_ff` vs actual `d_ff + emb`).
- **LOW [read]: 4-bit round-trip zeroes subnormal-magnitude rows.**
  Per-row scales persist as float32 (`tinylm/serialization.py:95`); a
  subnormal scale (e.g. `~1.4e-47`) flushes to `0.0` while nonzero nibbles
  survive, dequantizing to `[0.0, -0.0]`. Normal zeros/odd-length rows are fine.

## Router / runtime / router data — [exec]+[read]

- **HIGH [exec]: router `logits()` crashes on OOV IDs while `fast_logits()` clamps.**
  `_embed_context` indexes `self.C[tok]` directly (`tinylm/router.py:108-111`);
  after tokenizer growth, training crashes with `IndexError` while inference
  silently substitutes token 0 (`:153-155`). Train/infer inconsistency.
- **HIGH [read]: `decompose()` SEP becomes unreachable after vocab growth.**
  `SEP = len(tokenizer.vocab)-1` but `fast_logits` returns
  `config.vocab_size` entries (`tinylm/router.py:332-349`); once the tokenizer
  outgrows the router head, greedy `nxt` can never equal `SEP` — zero segments,
  full `max_tokens` always burned.
- **MEDIUM [read]: `route()` computes logits then never uses them.**
  `logits = self.fast_logits(context)` (`tinylm/router.py:234-235`) has no
  code use below (scoring is keywords-only; `:238` merely comments the idea).
  A wasted full forward pass per call plus a misleading contract.
- **HIGH [read]: `unload_all()` leaves ghost access entries, defeating eviction.**
  Clears `_models`/`_model_configs` but not `_model_last_access`
  (`tinylm/runtime.py:134-137`); `_unload_idle_models` evicts oldest-first
  from that dict (`:77-82`), so ghosts absorb the eviction slots while excess
  real models stay loaded (dict also grows unboundedly). Sibling:
  `unload_model()` never removes `_model_configs[name]` (`:126-132`, LOW leak).
- **MEDIUM [read]: TTL eviction races with serving threads.**
  The cleanup thread iterates `_model_last_access` (`tinylm/runtime.py:70`)
  while serving mutates it (`_mark_accessed`, `:86`); no `Lock` anywhere in
  the file → intermittent `RuntimeError: dictionary changed size`.
- **HIGH [read]: 8 encode slots collapse to 4 decode names.**
  Unknown models hash to slots 0–7 (`tinylm/router_data.py:26-28`) but decode
  mods by 4 (`:128-136`), so slots 4–7 alias to wrong fixed names — and every
  unknown name decodes to a wrong fixed name regardless. Silent routing-label
  corruption on round-trip.
- **HIGH [read]: `train_router()` can grow the tokenizer past the router head.**
  `_ensure_vocab_size` extends to `min_vocab + 20`
  (`tinylm/router_data.py:44-60`); with `min_vocab = len(router.C)`
  (`:187-189`), `SEP`/`MODEL_BASE` (`:76-77`) land past `len(router.W_out)`,
  and training indexes the head out of range (`IndexError`).
- **MEDIUM [read]: one corrupt JSONL line destroys the whole load.**
  `load_router_training_data` (`tinylm/training_data.py:66-82`) has no
  per-line guard: a single bad line raises and the caller gets zero examples.
- **MEDIUM [read]: `split_router_training_data()` pollutes the model namespace.**
  Splits are saved as `{source}_train`/`{source}_val`
  (`tinylm/training_data.py:109-125`), which match the
  `*_router_training.jsonl` glob, so `list_model_routers()` reports phantom
  models; `split_ratio` is unvalidated (1.5/negative/empty-train all silently accepted).
- **LOW [read]: bare-word text chunks mistranslated as markers.**
  `encode_marked` (`tinylm/data_clean.py:334-339`) maps any chunk equal to
  `EOL`/`EOS`/`EOT` to a marker ID — including literal text between real
  markers. Also `marker_id()` (`:290-294`) has no small-vocab guard:
  `vocab_size=10` yields `-3`, which silently negative-indexes later; and
  `generate_router_training_data` (`tinylm/router_data.py:234-236`) drops
  sentence terminators via `re.split` (training-text corruption).

## Generation / checkpoint / stats / schedulers — [exec]+[read]

- **HIGH [exec]: repetition penalty crashes on OOV ids in history.**
  `_sample_next_token` (`tinylm/generation.py:54-57`) indexes `logits[idx]`
  with no bounds check; ids ≥ model head (after vocab growth) → `IndexError`.
- **MEDIUM [exec]: penalty backfires on negative logits.**
  Same lines divide regardless of sign: `-2.0 / 2.0 = -1.0` moves toward zero,
  raising P(repeat) `0.119 → 0.269` (observed). Correct form scales positives
  down and negatives further negative.
- **MEDIUM [exec+read]: `temperature=0` crashes; negative inverts.**
  `1.0 / temperature` (`tinylm/generation.py:60-63`) → `ZeroDivisionError`
  for the standard greedy request; negative values silently invert the
  distribution instead of being rejected.
- **MEDIUM [read]: `stream_callback` receives `max_tokens`, not totals.**
  Docstring promises `(token_str, token_idx, total_generated, tokens_per_sec)`
  (`tinylm/generation.py:169-170`) but the call passes
  `(token_str, count, max_tokens, tps)` (`:237`) — progress displays get
  `None`/cap. Same region (`:236-242`): a `False` return ("stop early") is
  gated on `min_tokens`, so early-stop and stop-string hits are ignored until
  `min_tokens` tokens exist.
- **LOW [read]: negative context ids wrap instead of clamping.**
  `fast_logits` (`tinylm/generation.py:17-20`) clamps only `tok >= vocab`
  to `0`; `-1` wraps via `C[-1]` to the last row — silent wrong embedding.
- **HIGH [exec]: `resume_from_checkpoint` silently corrupts on shape mismatch.**
  `tinylm/checkpoint.py:57-60` zips without length/arch checks and ignores
  `checkpoint.config`: short state leaves tail rows stale, long state drops
  extras (observed 3-param model + 1 value → `[9.0, 2.0, 3.0]`, no error).
  Cross-arch or pre/post-growth resumes corrupt weights silently (contrast
  `load_best`, which refuses).
- **HIGH [exec]: NaN best-loss permanently poisons training stats.**
  `record()` stores non-finite `best_loss` (`tinylm/stats.py:53-68`); worse,
  `apply_stats` treats NaN-vs-None as improvement
  (`tinylm/stats.py:131-135`) and snapshots NaN weights as "best" — observed:
  `improved=True`, snapshot saved, and the next finite run (`0.5`) reports
  `improved=False, reverted=True` back to NaN weights. Permanent until manual
  file deletion.
- **MEDIUM [read]: corrupt `training_stats.json` is silently wiped.**
  `_load` (`tinylm/stats.py:31-38`) treats any parse failure as empty, and the
  next `record()` overwrites the file — history destroyed, no backup/warning.
  Sibling (`:140-144`): the revert path calls `load_model` unguarded with no
  param-count check, unlike `load_best` (`:90-99`), so a corrupt/bigger best
  file crashes mid-training or prefix-copies while reporting `reverted=True`.
- **MEDIUM [read]: `evaluate()` wastes a full backward per batch, leaves dirty grads.**
  `tinylm/sched.py:304-312` zeroes, back-props, then uses only `loss.data`;
  the last batch's grads stay dirty (benign in-tree — callers zero first —
  but pure waste on every eval).
- **MEDIUM [read]: cyclic LR snaps the multiplier back to 1.0 each cycle.**
  The triangle wave is anchored at 1.0 (`tinylm/sched.py:156-164`), undoing
  plateau reductions at every cycle boundary; the floor term divides a
  learning rate by a loss (`min_lr / best_loss`, wrong units, dead weight).
- **MEDIUM [read]: `EarlyStopping` restores only the param prefix after growth.**
  `best_params` has no length recorded/checked (`tinylm/sched.py:340-354`);
  if `sync_vocab` adds rows between, `zip` restores the prefix while new rows
  keep diverged values — reported as a full best-restore.
- **LOW [exec]: negative `max_norm` flips gradient signs.**
  `clip_grads` (`tinylm/sched.py:44-56`) accepts `max_norm=-1.0` and scales by
  a negative coefficient (`[1,1]` → `[-0.707,-0.707]`, observed). In-repo
  callers guard with `> 0`; direct callers get sign-flipped updates.
- **LOW [read]: Muon partial-index steps leak grads.**
  Skipped groups are neither stepped nor zeroed (`tinylm/sched.py:759-788`);
  latent only — no in-repo caller passes partial `indices` (`training.py:1210`,
  `parallel.py:477` use full steps, where params are overwritten).

## Harness / autotrain / CLI — [exec]+[read]

- **HIGH [exec]: autotraining can never succeed (`workers` kwarg).**
  `autotrain.py:392` calls `train(..., workers=1)` but `train()` has no such
  parameter (`tinylm/training.py:710`) → `TypeError` on every run, swallowed by
  the surrounding try into `{"success": False}`. Observed directly.
- **HIGH [read]: autotrain worker thread dies silently on load failures.**
  `_worker_loop` (`autotrain.py:426-443`) has no `try`, and
  `loader.load_model` (`:367-368`), tokenizer extend/sync (`:380-381`) sit
  outside `_train_model`'s inner try (`:391-424`). An unknown model raises
  through the daemon thread while `_running` stays `True` — status lies forever.
  Sibling: a failed batch is popped (`:436`) and only logged, never requeued,
  while dedup blocks those prompts from re-entering — silent training loss.
- **HIGH [read]: feedback file rewrite races with collection.**
  `_update_feedback_entry` (`autotrain.py:547-569`) does unlocked
  read-modify-rewrite while `collect()` appends under `_lock` (`:96-99`); a
  concurrent collect landing between read and `"w"`-rewrite is lost forever,
  and corrupt lines are dropped-then-rewritten (loss made permanent). Same
  pattern: `_pending` is appended from the feedback callback (`:300-302`)
  and popped by the worker (`:430-436`) with no shared lock.
- **HIGH [exec]: sandbox escape via prefix check.**
  `_resolve_path` (`scratchpad.py:638-644`) uses
  `str(target).startswith(str(root))`: root `/tmp/rt_root` admits
  `/tmp/rt_root2/evil.txt` (observed returned, not raised).
- **HIGH [exec]: `assert`/`branch` eval bypasses privileges.**
  `_cmd_assert` (`scratchpad.py:677-686`, same in `_cmd_branch` `:700-712`)
  runs `eval` with no privilege check, so default READ_ONLY executes arbitrary
  Python (observed subclass introspection returning `True`), bypassing the
  FULL_ACCESS gate on `code_run`/`shell`.
- **MEDIUM [read]: custom catalog silently bypassed in train paths.**
  `AutoTrainer` accepts `catalog` (`autotrain.py:243-245`, used at `:304`)
  but `_train_model`/`train_avoidance` hardcode global `MODEL_CATALOG` for
  load and metrics (`:367-368,402-409`, same at `:592-627`).
- **MEDIUM [exec+read]: CLI has no TRAIN/EMBED executors; bare/error paths exit 0.**
  `_default_task_executors` (`main.py:566-567`) covers 6 of 8 `TaskType`s —
  TRAIN/EMBED tasks always record FAILED. Bare `tinylm model` (and
  `autotrain`/`scratchpad`/`taskpad`, `main.py:265-276`+) prints nothing and
  exits 0 (observed); error paths (`taskpad execute no_such_task`,
  `main.py:599-608`) print `Error:` yet return normally → exit 0 (observed).
- **MEDIUM [read]: `parallel` returns completion order, races shared state.**
  `as_completed` pairing is nondeterministic vs command order and threads
  share unsynchronised `self.variables` (`scratchpad.py:714-722`).
- **LOW [read]: `$var` interpolation undocumentedly unsupported.**
  Docstring promises `${var}` or `$var` but only `${var}` is substituted
  (`scratchpad.py:646-652`).

## UI (Tk `uitk/`, Qt `uiqt/`) — [read]

- **HIGH [read]: Qt Save Config crashes on missing method.**
  `uiqt/act_settings.py:296-298` calls `self._load_gen_settings()`, defined
  nowhere repo-wide (single grep hit) → `AttributeError` after saving.
- **HIGH [read]: Tk Save-as-Preset crashes on missing import.**
  `uitk/act_settings.py:140` uses `tk.simpledialog` but the module imports
  only `tkinter as tk` plus `scrolledtext/messagebox/ttk/filedailog`
  (`:7-8`); bare `tk` never exposes the `simpledialog` submodule.
- **HIGH [read]: training always overwrites the single `tinyLM` slot.**
  Both `uitk/result.py:99` and `uiqt/result.py:146` unconditionally
  `save_model(self.model, MODEL_PATH)` after training any selected catalog
  model — foreign (possibly different-arch) weights clobber the TinyLM file,
  later served silently when vocab sizes match.
- **MEDIUM [read]: Tk cannot train RWKV (no routing).**
  The Tk job unconditionally calls `train_parallel`
  (`uitk/act_training.py:157`), which raises `TypeError` for non-`TinyLM`
  (`tinylm/parallel.py:287-290`). Qt routes RWKV to `train()`; `uitk/` has no
  `is_rwkv` branch anywhere (verified by grep).
- **MEDIUM [read]: switching back to TinyLM builds a random model.**
  Both `_load_model_job`s (`uitk/core.py:129-131`, `uiqt/core.py:157-159`)
  construct a fresh `TinyLM(...)` instead of loading the `tinyLM` file —
  garbage output with no indication despite trained weights on disk.
- **MEDIUM [read]: Tk model dropdown desyncs mid-training.**
  `on_model_change` overwrites `_model_name` before `_switch_model`'s busy
  abort (`uitk/core.py:112-122`); Qt guards `busy` first
  (`uiqt/core.py:184-186`). Later snapshots/labels/attribution use a name that
  was never loaded.
- **MEDIUM [read]: Tk router-training progress is dropped.**
  Router jobs post `("progress", "Router Training", ...)`
  (`uitk/act_training.py:243`) but the worker only handles label `"Training"`
  (`uitk/worker.py:86-92`) — bar frozen at 0 for the whole run (looks hung).
- **LOW [read]: valid config shows an error popup.**
  `uiqt/act_settings.py:105-117` reports success via `show_error`
  (`QMessageBox.critical`) — red alarm saying the config is valid.
- **LOW [read]: one bad generation field discards all fields silently.**
  `uitk/act_chat.py:55-58` (same at `:177-180`): any `ValueError` resets every
  setting to defaults with no warning.
- **LOW [read]: Qt kernel indicator mislabels RWKV + non-SGD.**
  The pre-flight tuple hardcodes the optimizer check to `False`
  (`uiqt/act_training.py:119-145`), so RWKV + adam + auto shows "CUDA (auto)"
  while `train()` falls back to Python.

## Training kernels / plumbing (prior entries, kept)

- **HIGH: `sync_vocab` crashes on RWKVModel — see Config section pattern.**
  `tinylm/serialization.py:11-28` hardcodes `b1/W1/W2/b2` → `AttributeError`
  on RWKV ([exec]); breaks growth in `uitk/act_training.py:116`,
  `uiqt/act_training.py:176`, `uiqt/act_settings.py:327`, `main.py:426`,
  `autotrain.py:380,609`, `tinylm/__main__.py:23`.
- **HIGH: legacy CUDA `train_step` silently ignores `grad_clip`.**
  `max_norm` appears only in signatures/pass-through
  (`csrc/train_kernel_cuda.cu:193,349,397`; `accel_cuda.py:108-166`, no host
  clip) — unlike `cuda_batch_grads`, the C kernel, and Python.
- **MEDIUM: `train_parallel`/`train_multicore` missing short-corpus guard.**
  `tinylm/parallel.py:299,413` and `:560,594,597`: no `len(ids) <= ctx`
  `ValueError`; `random.choice` on an empty range raises bare `IndexError`.
- **MEDIUM: cross-block race in GPU grad-zeroing.**
  Grid-stride zero + block-local `__syncthreads()` in
  `csrc/train_kernel_cuda.cu:199-207` and `csrc/rwkv_kernel.cu:532-537`;
  `batch > 256` lets accumulation interleave with zeroing (fix: host memset).
- **MEDIUM: `dot`/`matvec_dot` truncate on dim mismatch.**
  `tinylm/autograd.py:107-152` (`map`/`zip` stop short) instead of raising —
  masked the old RWKV `sampled_loss` bug.
- **LOW: dead `cuda_train_steps` with missing CDLL binding.**
  `accel_cuda.py:224-291`, no callers; `_load_lib` (`:26-80`) never declares
  it (seed would truncate to `c_int`, among others).
- **LOW: `gradient_accumulation` ignored on the C path.**
  Accepted at `tinylm/training.py:184`, unreferenced in `:176-425`.
- **LOW: parent clip misses `clip_grads`' `+1e-6`.**
  `tinylm/parallel.py:466-469` vs `tinylm/sched.py:44-56` (~1e-6 relative
  train-vs-parallel divergence when clipping).
- **LOW: build helpers reference missing attrs.**
  `build_cuda.sh:43` (`accel_cuda.CUDA_AVAILABLE`), `build_rwkv.sh:45`
  (`accel_rwkv.RW_KV_AVAILABLE`) — neither exists; self-test fails post-compile
  under `set -e`. (Canonical `./build.sh` unaffected.)
- **DOCS: duplicate backlog bullets** at `ROADMAP.md:86-87` (identical
  presampling entries).
