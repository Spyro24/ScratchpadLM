# BUGS.md — known bugs in ScratchpadLM

> Every entry below was verified by reading the cited code (and, where noted,
> by executing a reproducer). Severity: HIGH = crash / silent wrong training,
> MEDIUM = wrong behavior in a reachable path, LOW = cosmetic / latent / narrow.
> Line numbers refer to the current tree (commit `c2810f5`).

## HIGH

### 1. `sync_vocab` crashes on RWKVModel (vocab growth broken for RWKV)
- **Path:** `tinylm/serialization.py:11-28`
- The function hardcodes the legacy MLP layout: `len(model.b1)` (`:18`),
  `model.W1` (`:25`), `model.W2` (`:21,26`), `model.b2` (`:22,26`).
  `RWKVModel` has none of these (`C`, `in_proj`, `blocks`, `out_proj`,
  `out_bias` only), so any call raises
  `AttributeError: 'RWKVModel' object has no attribute 'b1'` (reproduced).
- **Impact:** every vocab-growth path crashes for RWKV models —
  `uitk/act_training.py:116`, `uiqt/act_training.py:176`,
  `uiqt/act_settings.py:327`, `main.py:426`, `autotrain.py:380,609`,
  `tinylm/__main__.py:23`. Violates the "never recreate, use `sync_vocab`"
  constraint for the canonical arch.

### 2. Legacy CUDA `train_step` silently ignores `grad_clip`
- **Path:** `csrc/train_kernel_cuda.cu:193,349,397`, `accel_cuda.py:108-166`
- `max_norm` is threaded through the signatures but never referenced in any
  device-code body (all 6 mentions are signatures/pass-through), and the
  `cuda_train_step` Python wrapper applies no host-side clip either —
  unlike `cuda_batch_grads` (host clip), the C kernel (`accel.py:120-143`),
  and the Python path (`clip_grads`). The training contract says global-norm
  clipping is always applied when set.
- **Impact:** `train(..., use_c="cuda", grad_clip=X)` on a legacy model
  trains unclipped while reporting the clip as active.

## MEDIUM

### 3. Tk UI cannot train RWKV models at all
- **Path:** `uitk/act_training.py:157`, `tinylm/parallel.py:287-290`
- The Tk training job unconditionally calls `train_parallel`, which raises
  `TypeError` for any non-`TinyLM` model. The Qt UI routes RWKV to `train()`
  (`uiqt/act_training.py:218-225`); `uitk/` has no `is_rwkv` branch anywhere.
- **Impact:** training an RWKV model from the Tk UI always errors out
  (loud, but a dead feature).

### 4. `train_parallel` / `train_multicore` accept too-short corpora, then crash obscurely
- **Path:** `tinylm/parallel.py:299,413` and `tinylm/parallel.py:560,594,597`
- Neither builds the `len(ids) <= ctx` → `ValueError` guard that `train()`
  and `_train_cuda` have. `starts` becomes an empty range and
  `random.choice(starts)` raises a bare `IndexError`.
- (Already noted at `ROADMAP.md:116`; listed here as the actual defect.)

### 5. Cross-block race between grad-zeroing and accumulation in GPU kernels
- **Path:** `csrc/train_kernel_cuda.cu:199-207`, `csrc/rwkv_kernel.cu:532-537`
- Both batch kernels zero `out_grads` with a grid-stride loop followed by a
  block-local `__syncthreads()`. With `batch > 256` (`nblocks > 1`) one block
  can start atomically accumulating window grads while another block is still
  zeroing → silently lost gradients.
- **Impact:** narrow (default/test batches are ≤ 32), but a real race.
  Robust fix: `cudaMemset` the grad buffer on the host before launch.

### 6. `dot` / `matvec_dot` silently truncate on dimension mismatch
- **Path:** `tinylm/autograd.py:107-152`
- Forward uses `sum(map(operator.mul, wd, xd))` and backward uses `zip(...)`,
  both of which stop at the shorter side instead of raising. A shape bug
  therefore computes wrong values/grads silently.
- **Impact:** this masked the old `RWKVModel.sampled_loss` bug (full
  `vocab`-vector dotted through `hidden`-rows) instead of failing fast.

## LOW

### 7. Build-helper self-tests reference attributes that do not exist
- **Path:** `build_cuda.sh:43`, `build_rwkv.sh:45`
- They print `accel_cuda.CUDA_AVAILABLE` / `accel_rwkv.RW_KV_AVAILABLE`,
  but the modules only define `_cuda_available()` / `_rwkv_available()`
  functions (cf. `accel_cuda.py:26-80`, `accel_rwkv.py`). The self-test raises
  `AttributeError`, and `set -e` then fails the script after a good compile.
  (The canonical `./build.sh` does not run these snippets, so normal builds
  are unaffected.)

### 8. Dead `cuda_train_steps` API with missing CDLL binding
- **Path:** `accel_cuda.py:224-291`
- No caller anywhere in the repo. Additionally `_load_lib`
  (`accel_cuda.py:26-80`) never declares its `argtypes`/`restype`, so calling
  it would go through ctypes defaults (e.g. the `c_ulonglong` seed truncated
  to `c_int`).

### 9. `gradient_accumulation` accepted but ignored on the C kernel path
- **Path:** `tinylm/training.py:184` (accepted; body `176-425` never references it)
- `_train_c` runs the fused per-batch step regardless, so
  `gradient_accumulation > 1` silently does nothing there (Python path honors it).
- (Already noted at `ROADMAP.md:89`; listed here as the actual defect.)

### 10. Parent clip in `train_parallel` misses the `+1e-6` of `clip_grads`
- **Path:** `tinylm/parallel.py:466-469` vs `tinylm/sched.py:44-56`
- Parent uses `scale = clip_now / total_norm`; `clip_grads` uses
  `max_norm / (total_norm + 1e-6)`. `train()` and `train_parallel()` diverge
  ~1e-6 relative whenever clipping engages.

### 11. Qt kernel indicator mislabels RWKV + non-SGD runs
- **Path:** `uiqt/act_training.py:119-145`
- The pre-flight `advanced_features` tuple hardcodes the optimizer check to
  `False`, so RWKV + `adam` + auto shows "CUDA (auto)" while `train()`
  (correctly) falls back to Python. Cosmetic only.

## DOCS

### 12. Duplicate backlog bullets in ROADMAP
- **Path:** `ROADMAP.md:86-87`
- The "`train()` upfront negative presampling … → lazy chunked refill" entry
  appears twice, verbatim.
