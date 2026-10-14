"""Data-parallel training with persistent workers (stdlib multiprocessing)."""
import math
import random
from multiprocessing import Pool, cpu_count
from typing import List

try:
    import accel
except Exception:  # pragma: no cover
    accel = None

from tinylm.checkpoint import _flat_params, _set_flat_params
from tinylm.sched import (
    RollbackGuard, clip_grads, AutoAdjust, LR_Scheduler,
    AdamOptimizer, LionOptimizer, MuonOptimizer,
    APOLLOOptimizer, APOLLOMiniOptimizer,
)
from tinylm.training import (
    compute_token_frequencies, build_alias_table, alias_sample,
    n_negatives_schedule,
)

class _WorkerState:
    """Holds a model instance in each worker process to avoid reconstruction."""
    model = None
    cfg = None
    vocab_size = 0
    n_negatives = 0
    use_c = False
    use_gpu = False
    max_norm = 0.0


def _init_worker(flat, cfg, vocab_size, n_negatives, max_norm=0.0, use_gpu=False):
    """Initializer for persistent worker pool."""
    from tinylm.models_tiny import TinyLM
    from tinylm.training import _accel_available
    _WorkerState.cfg = cfg
    _WorkerState.vocab_size = vocab_size
    _WorkerState.n_negatives = n_negatives
    _WorkerState.max_norm = max_norm
    try:
        import accel_cuda
        _WorkerState.use_gpu = bool(use_gpu) and accel_cuda._cuda_available()
    except Exception:
        _WorkerState.use_gpu = False
    _WorkerState.use_c = _accel_available()
    if _WorkerState.use_c:
        # C kernel computes grads directly on flat floats — no model needed
        _WorkerState.model = None
        return
    _WorkerState.model = TinyLM(
        vocab_size=cfg["vocab"],
        ctx=cfg["ctx"],
        emb=cfg["emb"],
        hidden=cfg["hidden"],
    )
    _set_flat_params(_WorkerState.model, flat)


def _worker_grads_v2(args):
    """Worker function that receives (flat, windows[, n_neg]) and uses persistent model.

    Each window is (context, target) or (context, target, negs-or-None);
    pre-sampled negatives (frequency-weighted) are honored when present.
    The optional third task element overrides the pool-level n_negatives
    (adaptive schedule); the C path samples that many internally.
    """
    flat, windows = args[0], args[1]
    n_neg = args[2] if len(args) > 2 else _WorkerState.n_negatives
    if _WorkerState.use_gpu:
        # GPU grads on flat floats (no model needed); any failure falls back
        # to the C/Python paths below so a GPU hiccup never kills training.
        # Pre-sampled negatives must be uniform within the task (they are:
        # the parent samples current_n_negatives per window); otherwise let
        # the kernel sample internally.
        try:
            import accel_cuda
            cfg = _WorkerState.cfg
            dims = (cfg["vocab"], cfg["ctx"], cfg["emb"], cfg["hidden"])
            flat_windows = []
            flat_negs = []
            uniform_negs = True
            for w in windows:
                flat_windows.extend(w[0])
                flat_windows.append(w[1])
                if len(w) > 2 and w[2] is not None and len(w[2]) == n_neg:
                    flat_negs.extend(w[2])
                else:
                    uniform_negs = False
            return accel_cuda.cuda_batch_grads(
                flat, flat_windows, dims, n_neg,
                random.randrange(1 << 62),
                negatives=flat_negs if uniform_negs else None,
                max_norm=_WorkerState.max_norm)
        except Exception:
            pass
    if _WorkerState.use_c:
        cfg = _WorkerState.cfg
        dims = (cfg["vocab"], cfg["ctx"], cfg["emb"], cfg["hidden"])
        flat_windows = []
        for w in windows:
            context, target = w[0], w[1]
            flat_windows.extend(context)
            flat_windows.append(target)
        return accel.batch_loss_grads(flat, flat_windows, dims,
                                      n_neg,
                                      random.randrange(1 << 62),
                                      max_norm=_WorkerState.max_norm)
    model = _WorkerState.model
    _set_flat_params(model, flat)
    for p in model.params:
        p.grad = 0.0  # workers persist across steps; never carry grads over

    losses = []
    for w in windows:
        context, target = w[0], w[1]
        negs = w[2] if len(w) > 2 else None
        if negs is not None:
            losses.append(model.sampled_loss(context, target, len(negs), _WorkerState.vocab_size, negatives=negs))
        else:
            losses.append(model.sampled_loss(context, target, n_neg, _WorkerState.vocab_size))
    # Sum losses
    loss = losses[0]
    for L in losses[1:]:
        loss = loss + L
    loss.backward()
    # Apply gradient clipping in Python path
    if _WorkerState.max_norm > 0:
        clip_grads(model.params, _WorkerState.max_norm)
    return loss.data, [p.grad for p in model.params]


def _flat_windows(batch):
    """Flatten [(context, target), ...] to [ctx ids..., target] per window."""
    flat = []
    for context, target in batch:
        flat.extend(context)
        flat.append(target)
    return flat


def _pick_winner(results):
    """Index of the trial to adopt, or None when every trial is unusable.

    Each result is (shard_loss, probe_loss, flat, velocity). Primary rule:
    lowest finite probe loss on finite weights. Fallback: lowest finite
    shard loss on finite weights (probe can go non-finite while the trial
    itself is fine).
    """
    best_i, best_l = None, float('inf')
    for i, (shard_loss, probe_loss, flat, _vel) in enumerate(results):
        if (math.isfinite(probe_loss) and probe_loss < best_l
                and all(math.isfinite(x) for x in flat)):
            best_i, best_l = i, probe_loss
    if best_i is not None:
        return best_i
    for i, (shard_loss, _probe, flat, _vel) in enumerate(results):
        if (math.isfinite(shard_loss) and shard_loss < best_l
                and all(math.isfinite(x) for x in flat)):
            best_i, best_l = i, shard_loss
    return best_i


def _multicore_trial(args):
    """One full training trial on a worker: full-batch step + probe eval.

    args = (flat, velocity, shard, probe, lr, momentum, max_norm), all plain
    floats/ints (never `Value`). Returns (shard_loss, probe_loss, new_flat,
    new_velocity). The shard loss is the pre-update batch mean; the probe
    loss scores the updated weights on the parent's shared probe batch so
    trials are comparable.
    """
    flat, velocity, shard, probe, lr, momentum, max_norm = args
    n = len(shard)
    if _WorkerState.use_c:
        cfg = _WorkerState.cfg
        dims = (cfg["vocab"], cfg["ctx"], cfg["emb"], cfg["hidden"])
        loss_sum, grad_sum = accel.batch_loss_grads(
            flat, _flat_windows(shard), dims, _WorkerState.n_negatives,
            random.randrange(1 << 62))
        g = [x / n for x in grad_sum]
        if max_norm > 0:
            norm = math.sqrt(sum(x * x for x in g))
            if norm > max_norm and norm > 0:
                scale = max_norm / norm
                g = [x * scale for x in g]
        vel = [momentum * v + x for v, x in zip(velocity, g)]
        new_flat = [x - lr * v for x, v in zip(flat, vel)]
        probe_sum, _ = accel.batch_loss_grads(
            new_flat, _flat_windows(probe), dims, _WorkerState.n_negatives,
            random.randrange(1 << 62))
        return (loss_sum / n, probe_sum / len(probe), new_flat, vel)
    model = _WorkerState.model
    _set_flat_params(model, flat)
    for p in model.params:
        p.grad = 0.0
    losses = []
    for context, target in shard:
        losses.append(model.sampled_loss(context, target, _WorkerState.n_negatives,
                                         _WorkerState.vocab_size))
    loss = losses[0]
    for L in losses[1:]:
        loss = loss + L
    loss.backward()
    for p in model.params:
        p.grad /= n
    if max_norm > 0:
        clip_grads(model.params, max_norm)
    vel = [momentum * v + p.grad for v, p in zip(velocity, model.params)]
    new_flat = [x - lr * v for x, v in zip(flat, vel)]
    _set_flat_params(model, new_flat)
    probe_loss = sum(
        model.sampled_loss(c, t, _WorkerState.n_negatives,
                           _WorkerState.vocab_size).data
        for c, t in probe) / len(probe)
    return (loss.data / n, probe_loss, new_flat, vel)


def train_parallel(model, ids, steps=2000, lr=0.05, batch_size=16, n_negatives=256, workers=None,
                   progress_cb=None, grad_clip: float = 0.0, cancel_callback=None,
                   save_on_cancel_callback=None, momentum: float = 0.9,
                   rollback_window: int = 0, rollback_tol: float = 1e-4,
                   rollback_callback=None, auto_adjust: bool = True,
                   auto_factor: float = 0.5, auto_patience: int = 10,
                   auto_clip_relax: bool = True,
                   lr_scheduler: str = "linear", warmup_steps: int = 0,
                   min_lr: float = 0.0,
                   # New AutoAdjust features
                   use_cosine_restarts: bool = False, restart_period: int = 0,
                   cyclic_lr: bool = False, cycle_length: int = 0,
                   ema_alpha: float = 0.9,
                   use_frequency_neg_sampling: bool = False,
                   n_negatives_start: int = None, n_negatives_end: int = 64,
                   n_negatives_decay: str = "exponential",
                   # Optimizer selection (parent update only; workers return grads).
                   optimizer: str = "sgd",
                   weight_decay: float = 0.0,
                   adam_lr_scale: float = 0.5,
                   lion_lr_scale: float = 0.1,
                   muon_scale: float = 0.5,
                   muon_max_dim: int = 64,
                   muon_ns_steps: int = 5,
                   apollo_rank: int = 256,
                   apollo_beta1: float = 0.9,
                   apollo_beta2: float = 0.999,
                   apollo_eps: float = 1e-8,
                   apollo_proj_update_freq: int = 200,
                   norm_growth_limit: float = 1.01,
                   # Batched compute: accumulate grads over this many batches
                   # per update (larger effective batch, more GPU work per step).
                   batch_steps: int = 1,
                   # Route worker grads through the CUDA kernel when available
                   # (falls back per task to C/Python, never fails training).
                   use_gpu: bool = True):
    """Data-parallel version of train. Uses persistent worker pool to avoid
    rebuilding the model each step. The batch is split across processes and
    the gradients are summed before one SGD update (with momentum).

    Matches `train()`: gradients are averaged over the batch, clipped once by
    global norm, then applied as `v = momentum * v + g ; p -= lr * v`
    (`momentum=0` -> plain SGD). Automatic adjustment (`auto_adjust`: LR cut
    by `auto_factor` on plateaus + clip relaxation via `auto_clip_relax`)
    mirrors `train()` with `rollback_tol` as its `min_delta`.

    `rollback_window` snapshots every step and restores the lowest-loss step
    in the window when the current loss spikes above it by more than
    `rollback_tol` (0 disables). `rollback_callback(step, loss, best_loss)`
    fires on every rollback with the run's smallest loss so far.

    `progress_cb(done, total, loss, val_loss=None, lr=None, ...)` is called
    after every step if given (runs in the main process; safe for queue-based
    UI updates). Extra auto-adjust state (aa_mult, aa_momentum, aa_grad_clip,
    aa_reduce, aa_bad, aa_best) is passed as keywords, mirroring `train()`.
    `batch_steps` batches are computed per update (gradient accumulation);
    `use_gpu` routes worker gradients through the CUDA kernel when available.
    `cancel_callback()` - if provided and returns True, training stops early.
      On cancel: if best loss < initial loss the improved weights are kept
      (and `save_on_cancel_callback(model, best_loss, steps_done)` is called
      when provided); otherwise the model is restored to its initial weights.
    Returns a dict with `cancelled`, `initial_loss`, `best_loss`,
    `final_loss`, `steps_done`, `rollbacks`, `kernel_used`.
    """
    workers = workers or cpu_count()
    if type(model).__name__ != "TinyLM":
        raise TypeError(
            f"parallel training only supports the legacy TinyLM MLP arch, got {type(model).__name__}; "
            "use train() (Python path) for RWKVModel until its parallel/GPU kernel lands")
    if batch_steps < 1:
        raise ValueError("batch_steps must be >= 1")
    try:
        import accel_cuda
        gpu_available = accel_cuda._cuda_available()
    except Exception:
        gpu_available = False
    gpu_active = bool(use_gpu) and gpu_available
    ctx = model.ctx
    vocab = len(model.C)
    starts = range(len(ids) - ctx)
    cfg = {"vocab": vocab, "ctx": ctx, "emb": len(model.C[0]), "hidden": len(model.b1)}
    params = model.params

    # Snapshot initial weights so a cancelled run without improvement can be reverted.
    initial_params_snapshot = [p.data for p in params]

    # Track initial/best loss for cancel-and-save-if-improved feature
    initial_loss = None
    best_loss = float('inf')
    best_params_snapshot = None
    last_loss = None
    cancelled = False
    steps_done = 0
    velocity = [0.0] * len(params)
    rb_guard = RollbackGuard(window=rollback_window, tol=rollback_tol)

    # Automatic adjustment (LR on plateau + clip relaxation), as in train().
    adjuster = AutoAdjust(
        enabled=auto_adjust, factor=auto_factor, patience=auto_patience,
        min_delta=rollback_tol, warmup_ignore=warmup_steps,
        max_grad_clip=grad_clip, clip_relax=auto_clip_relax, momentum=momentum,
        # New AutoAdjust features
        use_cosine_restarts=use_cosine_restarts,
        restart_period=restart_period,
        cyclic_lr=cyclic_lr,
        cycle_length=cycle_length,
        ema_alpha=ema_alpha,
        min_lr=min_lr,
    )

    # LR schedule (mirrors train(); default linear decay for back-compat).
    scheduler = LR_Scheduler(lr, steps, lr_scheduler, warmup_steps, min_lr)

    # Frequency-based negative sampling (unigram^0.75, word2vec style).
    neg_alias, neg_prob = [], []
    if use_frequency_neg_sampling:
        freq_probs = compute_token_frequencies(ids, vocab)
        neg_alias, neg_prob = build_alias_table(freq_probs)
        print(f"Built frequency-based negative sampling (vocab={vocab}, {sum(1 for f in freq_probs if f > 0)} active tokens)")
    neg_rng = random.Random()

    def _aa_state():
        """Current auto-adjust state for progress callbacks."""
        return dict(aa_mult=adjuster.multiplier, aa_momentum=adjuster.momentum,
                    aa_grad_clip=adjuster.grad_clip_now(), aa_reduce=adjuster.reduce_count,
                    aa_bad=adjuster.bad_checks, aa_best=adjuster.best_loss)

    # Optimizer for the parent update (workers only return grads).
    opt = None
    if optimizer == "adam":
        opt = AdamOptimizer(params, lr=lr, weight_decay=weight_decay,
                            lr_scale=adam_lr_scale)
    elif optimizer == "lion":
        opt = LionOptimizer(params, lr=lr, weight_decay=weight_decay,
                            lr_scale=lion_lr_scale)
    elif optimizer == "muon":
        try:
            tensors = model.get_tensors()
        except Exception:
            tensors = None
        opt = MuonOptimizer(params, tensors=tensors, lr=lr, beta=momentum,
                            weight_decay=weight_decay, lr_scale=adam_lr_scale,
                            muon_scale=muon_scale, max_dim=muon_max_dim,
                            ns_steps=muon_ns_steps)
    elif optimizer == "apollo":
        opt = APOLLOOptimizer(params, lr=lr, rank=apollo_rank, beta1=apollo_beta1,
                              beta2=apollo_beta2, eps=apollo_eps, weight_decay=weight_decay,
                              proj_update_freq=apollo_proj_update_freq,
                              norm_growth_limit=norm_growth_limit, min_lr=min_lr)
    elif optimizer == "apollo_mini":
        opt = APOLLOMiniOptimizer(params, lr=lr, beta1=apollo_beta1, beta2=apollo_beta2,
                                  eps=apollo_eps, weight_decay=weight_decay,
                                  proj_update_freq=apollo_proj_update_freq,
                                  norm_growth_limit=norm_growth_limit, min_lr=min_lr)
    elif optimizer != "sgd":
        raise ValueError(f"Unknown optimizer for train_parallel: {optimizer} "
                         "(expected sgd/apollo/apollo_mini/adam/lion/muon)")

    # Pre-create initial flat params for worker initialization.
    # Workers only compute gradients: clipping is applied once in the parent
    # on the batch-averaged gradient, so the in-worker clip stays off.
    init_flat = _flat_params(model)

    with Pool(
        workers,
        initializer=_init_worker,
        initargs=(init_flat, cfg, vocab, n_negatives, 0.0, use_gpu),
    ) as pool:
        for step in range(steps):
            # Check for cancellation before starting the next step.
            if cancel_callback is not None and cancel_callback():
                cancelled = True
                steps_done = step
                break

            rate = scheduler.get_lr(step) * adjuster.multiplier

            # Batched compute: accumulate `batch_steps` batches before one
            # update (larger effective batch, more GPU work per step).
            flat = _flat_params(model)
            total_loss = 0.0
            grad = [0.0] * len(params)
            n_windows = 0
            for _ in range(batch_steps):
                # Adaptive n_negatives (mirrors train()).
                current_n_negatives = n_negatives
                if n_negatives_start is not None and n_negatives_start != n_negatives_end:
                    current_n_negatives = n_negatives_schedule(
                        step, steps, n_negatives_start, n_negatives_end, n_negatives_decay
                    )

                batch = []
                for _ in range(batch_size):
                    i = random.choice(starts)
                    context, target = ids[i:i + ctx], ids[i + ctx]
                    negs = None
                    if use_frequency_neg_sampling and neg_alias:
                        negs = [alias_sample(neg_alias, neg_prob, neg_rng) for _ in range(current_n_negatives)]
                    batch.append((context, target, negs))
                chunks = [batch[k::workers] for k in range(workers)]
                chunks = [c for c in chunks if c]

                # Send (flat, chunk, n_negatives) to each worker
                results = pool.map(_worker_grads_v2, [(flat, c, current_n_negatives) for c in chunks])

                for loss_sum, g in results:
                    total_loss += loss_sum
                    for k, x in enumerate(g):
                        grad[k] += x
                n_windows += batch_size
            # Average the summed gradients over all windows (one update per step)
            grad = [g / n_windows for g in grad]

            # Track initial and best loss (pre-update loss)
            step_loss = total_loss / n_windows
            if rb_guard.enabled:
                do_rb, best_entry = rb_guard.check(step_loss)
                if do_rb:
                    _, best_flat, best_vel = best_entry
                    _set_flat_params(model, list(best_flat))
                    if best_vel is not None:
                        velocity = list(best_vel)
                    rb_guard.note_rollback()
                    if initial_loss is None:
                        initial_loss = step_loss
                    last_loss = step_loss
                    steps_done = step + 1
                    print(f"step {step:5d}  rollback: loss {step_loss:.3f} > best {best_entry[0]:.3f} in window {rollback_window} -> restored")
                    if rollback_callback is not None and best_loss != float('inf'):
                        rollback_callback(step + 1, step_loss, best_loss)
                    if progress_cb is not None:
                        progress_cb(step + 1, steps, step_loss, None, rate, **_aa_state())
                    continue
                rb_guard.push(step_loss, _flat_params(model), velocity)
            if initial_loss is None:
                initial_loss = step_loss
            if step_loss < best_loss:
                best_loss = step_loss
                best_params_snapshot = [p.data for p in params]
            last_loss = step_loss
            steps_done = step + 1

            # Gradient clipping on the mean gradient (same rule as train()),
            # with automatic relaxation when clipping fires chronically.
            clip_now = adjuster.grad_clip_now()
            total_norm = math.sqrt(sum(g * g for g in grad))
            adjuster.note_grad_clip(total_norm)
            if clip_now > 0 and total_norm > clip_now and total_norm > 0:
                scale = clip_now / total_norm
                grad = [g * scale for g in grad]

            # SGD with momentum: v = momentum * v + g ; p -= rate * v
            # Optimizer step (Adam/Lion/Muon) or plain SGD with momentum.
            if opt is not None:
                for p, g in zip(params, grad):
                    p.grad = g
                opt.lr = rate
                opt.step()
            else:
                mom = adjuster.momentum
                for i, p in enumerate(params):
                    velocity[i] = mom * velocity[i] + grad[i]
                    p.data -= rate * velocity[i]

            # Automatic adjustment: reduce LR when loss plateaus
            if adjuster.check(step + 1, step_loss):
                print(f"step {step:5d}  auto-adjust: lr multiplier -> {adjuster.multiplier:.4f}  momentum -> {adjuster.momentum:.4f}")

            if progress_cb is not None:
                progress_cb(step + 1, steps, step_loss, None, rate, **_aa_state())

            if step % 25 == 0:
                print(f"step {step:5d}  loss {step_loss:.3f}")

    # Handle cancellation: keep improved weights, else revert to initial.
    if cancelled:
        improved = (
            initial_loss is not None
            and best_loss < initial_loss
            and best_params_snapshot is not None
        )
        if improved:
            for prm, val in zip(params, best_params_snapshot):
                prm.data = val
            if save_on_cancel_callback is not None:
                save_on_cancel_callback(model, best_loss, steps_done)
        else:
            for prm, val in zip(params, initial_params_snapshot):
                prm.data = val

    return {
        "cancelled": cancelled,
        "initial_loss": initial_loss,
        "best_loss": best_loss if best_loss != float('inf') else None,
        "final_loss": last_loss,
        "steps_done": steps_done,
        "rollbacks": rb_guard.rollbacks,
        # Workers compute grads with the GPU kernel when requested and
        # available, else the C kernel when available, else Python. The parent
        # update always runs in Python.
        "kernel_used": ("parallel/GPU" if gpu_active
                        else ("parallel/C" if accel is not None and getattr(accel, "AVAILABLE", False)
                              else "parallel/Python")),
    }


def train_multicore(model, ids, steps=2000, lr=0.05, batch_size=16, n_negatives=256, workers=None,
                    progress_cb=None, grad_clip: float = 0.0, cancel_callback=None,
                    save_on_cancel_callback=None, momentum: float = 0.9,
                    rollback_window: int = 0, rollback_tol: float = 1e-4,
                    rollback_callback=None):
    """Best-of-N multicore training: one full step per core, keep the best.

    Each step every worker runs one complete training step (a full
    `batch_size`-window batch with SGD momentum) from the shared weights on
    its own random shard, then scores its updated weights on a shared probe
    batch sampled by the parent. The trial with the lowest probe loss wins
    and its weights + velocity are adopted, so trials are comparable (raw
    step losses on different shards would not be). Wall-clock per step is
    roughly one full step on N cores at the cost of N x compute, and keeping
    the best of N directions reduces loss faster per step than a single trial.

    Matches `train_parallel()`: `v = momentum * v + g ; p -= rate * v` with a
    linear LR decay, gradient clipping on the mean gradient, cancel/rollback
    semantics, and `progress_cb(done, total, loss)` reporting the adopted
    trial's pre-update loss.
    `cancel_callback()` - if provided and returns True, training stops early.
      On cancel: if best loss < initial loss the improved weights are kept
      (and `save_on_cancel_callback(model, best_loss, steps_done)` is called
      when provided); otherwise the model is restored to its initial weights.
    Returns a dict with `cancelled`, `initial_loss`, `best_loss`,
    `final_loss`, `steps_done`, `rollbacks`.
    """
    workers = workers or cpu_count()
    if type(model).__name__ != "TinyLM":
        raise TypeError(
            f"parallel training only supports the legacy TinyLM MLP arch, got {type(model).__name__}; "
            "use train() (Python path) for RWKVModel until its parallel/GPU kernel lands")
    ctx = model.ctx
    vocab = len(model.C)
    starts = range(len(ids) - ctx)
    cfg = {"vocab": vocab, "ctx": ctx, "emb": len(model.C[0]), "hidden": len(model.b1)}
    params = model.params

    # Snapshot initial weights so a cancelled run without improvement can be reverted.
    initial_params_snapshot = [p.data for p in params]

    # Track initial/best loss for cancel-and-save-if-improved feature
    initial_loss = None
    best_loss = float('inf')
    best_params_snapshot = None
    last_loss = None
    cancelled = False
    steps_done = 0
    velocity = [0.0] * len(params)
    rb_guard = RollbackGuard(window=rollback_window, tol=rollback_tol)

    init_flat = _flat_params(model)

    with Pool(
        workers,
        initializer=_init_worker,
        initargs=(init_flat, cfg, vocab, n_negatives, 0.0),
    ) as pool:
        for step in range(steps):
            # Check for cancellation before starting the next step.
            if cancel_callback is not None and cancel_callback():
                cancelled = True
                steps_done = step
                break

            rate = lr * (1.0 - step / steps)
            # One full shard per worker plus one shared probe batch.
            shards = [[(ids[i:i + ctx], ids[i + ctx])
                       for i in (random.choice(starts) for _ in range(batch_size))]
                      for _ in range(workers)]
            probe = [(ids[i:i + ctx], ids[i + ctx])
                     for i in (random.choice(starts) for _ in range(batch_size))]

            flat = _flat_params(model)
            results = pool.map(
                _multicore_trial,
                [(flat, list(velocity), s, probe, rate, momentum, grad_clip)
                 for s in shards])
            wi = _pick_winner(results)
            if wi is None:
                # Every trial diverged. Fall back to the rollback snapshot when
                # available; otherwise keep current weights and move on.
                do_rb, best_entry = rb_guard.check(float("nan")) if rb_guard.enabled else (False, None)
                if do_rb:
                    _, best_flat, best_vel = best_entry
                    _set_flat_params(model, list(best_flat))
                    if best_vel is not None:
                        velocity = list(best_vel)
                    rb_guard.note_rollback()
                    if initial_loss is None:
                        initial_loss = float("nan")
                    last_loss = float("nan")
                    steps_done = step + 1
                    print(f"step {step:5d}  rollback: all {workers} trials non-finite -> restored best {best_entry[0]:.3f}")
                    if rollback_callback is not None and best_loss != float('inf'):
                        rollback_callback(step + 1, float("nan"), best_loss)
                    if progress_cb is not None:
                        progress_cb(step + 1, steps, last_loss)
                    continue
                print(f"step {step:5d}  multicore: all {workers} trials non-finite -> kept weights")
                last_loss = float("nan")
                steps_done = step + 1
                if progress_cb is not None:
                    progress_cb(step + 1, steps, last_loss)
                continue
            step_loss, _probe_loss, new_flat, new_vel = results[wi]

            # Loss-spike rollback on the adopted trial's pre-update loss.
            if rb_guard.enabled:
                do_rb, best_entry = rb_guard.check(step_loss)
                if do_rb:
                    _, best_flat, best_vel = best_entry
                    _set_flat_params(model, list(best_flat))
                    if best_vel is not None:
                        velocity = list(best_vel)
                    rb_guard.note_rollback()
                    if initial_loss is None:
                        initial_loss = step_loss
                    last_loss = step_loss
                    steps_done = step + 1
                    print(f"step {step:5d}  rollback: loss {step_loss:.3f} > best {best_entry[0]:.3f} in window {rollback_window} -> restored")
                    if rollback_callback is not None and best_loss != float('inf'):
                        rollback_callback(step + 1, step_loss, best_loss)
                    if progress_cb is not None:
                        progress_cb(step + 1, steps, step_loss)
                    continue
                rb_guard.push(step_loss, _flat_params(model), velocity)

            # Track initial and best loss (pre-update loss)
            if initial_loss is None:
                initial_loss = step_loss
            if step_loss < best_loss:
                best_loss = step_loss
                best_params_snapshot = [p.data for p in params]
            # Adopt the winning trial's weights + velocity (in place).
            for p, x in zip(params, new_flat):
                p.data = x
            velocity = list(new_vel)
            last_loss = step_loss
            steps_done = step + 1

            if progress_cb is not None:
                progress_cb(step + 1, steps, step_loss)

            if step % 25 == 0:
                print(f"step {step:5d}  loss {step_loss:.3f}  winner {wi + 1}/{workers}")

    # Handle cancellation: keep improved weights, else revert to initial.
    if cancelled:
        improved = (
            initial_loss is not None
            and best_loss < initial_loss
            and best_params_snapshot is not None
        )
        if improved:
            for prm, val in zip(params, best_params_snapshot):
                prm.data = val
            if save_on_cancel_callback is not None:
                save_on_cancel_callback(model, best_loss, steps_done)
        else:
            for prm, val in zip(params, initial_params_snapshot):
                prm.data = val

    return {
        "cancelled": cancelled,
        "initial_loss": initial_loss,
        "best_loss": best_loss if best_loss != float('inf') else None,
        "final_loss": last_loss,
        "steps_done": steps_done,
        "rollbacks": rb_guard.rollbacks,
    }


class _PopState:
    """Per-worker static data for population trials (set once by initializer)."""
    arch = None
    config = None
    ids = None
    kwargs = None


def _init_population(arch, config, ids, kwargs):
    """Pool initializer: stash static trial data in each worker process."""
    _PopState.arch = arch
    _PopState.config = config
    _PopState.ids = ids
    _PopState.kwargs = kwargs


def _population_trial(args):
    """Run one independent replica for K steps from given weights.

    args = (flat, seed, steps). Reconstructs the model from (arch, config),
    trains with `train()` (single-process path, CPU or C/CUDA per `use_c`),
    and returns (best_loss, final_loss, new_flat, steps_done, rollbacks).
    Closures/callables can't cross processes, so no callbacks run in here.
    """
    flat, seed, steps = args
    from tinylm.training import train
    from tinylm.models_tiny import TinyLM
    try:
        from tinylm.models_linear import RWKVModel
    except Exception:
        RWKVModel = None
    cfg = dict(_PopState.config)
    cfg.pop("arch", None)
    if _PopState.arch == "RWKVModel" and RWKVModel is not None:
        model = RWKVModel(**cfg)
    else:
        model = TinyLM(**cfg)
    for p, v in zip(model.params, flat):
        p.data = v
    random.seed(seed)
    kw = dict(_PopState.kwargs)
    kw["steps"] = steps
    res = train(model, _PopState.ids, progress_cb=None, cancel_callback=None,
                **kw)
    best = res.get("best_loss")
    return (best if best is not None else float('inf'),
            res.get("final_loss"), [p.data for p in model.params],
            res.get("steps_done", 0), res.get("rollbacks", 0))


def train_population(model, ids, steps=200, replicas=8, select_every=10,
                     workers=None, lr=0.1, batch_size=32, n_negatives=256,
                     lr_scheduler: str = "cosine_warmup", warmup_steps: int = 50,
                     min_lr: float = 1e-5, grad_clip: float = 1.0,
                     momentum: float = 0.95, optimizer: str = "sgd",
                     weight_decay: float = 0.0,
                     adam_lr_scale: float = 0.5, lion_lr_scale: float = 0.1,
                     muon_scale: float = 0.5, muon_max_dim: int = 64,
                     muon_ns_steps: int = 5,
                     auto_adjust: bool = True, auto_factor: float = 0.7,
                     auto_patience: int = 5, auto_clip_relax: bool = True,
                     use_cosine_restarts: bool = False, restart_period: int = 0,
                     cyclic_lr: bool = False, cycle_length: int = 0,
                     use_frequency_neg_sampling: bool = False,
                     n_negatives_start: int = None, n_negatives_end: int = 64,
                     n_negatives_decay: str = "cosine",
                     rollback_window: int = 0, rollback_tol: float = 1e-4,
                     progress_cb=None, cancel_callback=None,
                     save_on_cancel_callback=None, use_c=False, seed: int = 0):
    """Population training: N replicas, keep the best every K steps.

    Each generation every replica trains independently for `select_every`
    steps from the same broadcast weights (different RNG seeds, so trajectories
    diverge). The replica with the lowest best-loss wins and its weights are
    broadcast to all replicas for the next generation (elitist selection).
    Optimizer state is reset at each broadcast — moments from different
    trajectories are not comparable.

    This parallelizes across CPU cores (one process per replica). It does NOT
    by itself load the GPU: per-replica compute follows `use_c` (`False` =
    pure Python, `"c"`/ `"cuda"` = legacy-MLP kernels where available).
    RWKVModel replicas use the Python path.

    Tip: keep `select_every` long enough (>= ~20 steps) that momentum and the
    LR schedule pay off inside a generation; very short generations waste the
    per-generation optimizer-state reset and underperform a single run.

    `progress_cb(done, total, loss, val_loss=None, lr=None)` reports the
    winning replica's best loss per generation. Same result contract as
    `train_parallel()` plus `generations` and winning-replica info.
    """
    import math as _math
    workers = workers or min(max(1, replicas), cpu_count())
    ctx = model.ctx
    if len(ids) <= ctx:
        raise ValueError(
            f"Not enough training tokens ({len(ids)}) for ctx={ctx}: "
            "need at least ctx+1 tokens"
        )
    if replicas < 1:
        raise ValueError("replicas must be >= 1")
    if select_every < 1:
        raise ValueError("select_every must be >= 1")
    arch = type(model).__name__
    config = model.get_config()
    params = model.params

    initial_params_snapshot = [p.data for p in params]
    broadcast_flat = list(initial_params_snapshot)

    worker_kwargs = dict(
        lr=lr, batch_size=batch_size, n_negatives=n_negatives,
        lr_scheduler=lr_scheduler, warmup_steps=warmup_steps, min_lr=min_lr,
        grad_clip=grad_clip, val_split=0.0, eval_every=0,
        early_stopping=False, patience=10, min_delta=rollback_tol,
        checkpoint_every=0, auto_adjust=auto_adjust, auto_factor=auto_factor,
        auto_patience=auto_patience, auto_clip_relax=auto_clip_relax,
        gradient_accumulation=1, momentum=momentum,
        rollback_window=rollback_window, optimizer=optimizer,
        weight_decay=weight_decay, adam_lr_scale=adam_lr_scale,
        lion_lr_scale=lion_lr_scale, muon_scale=muon_scale,
        muon_max_dim=muon_max_dim, muon_ns_steps=muon_ns_steps,
        use_frequency_neg_sampling=use_frequency_neg_sampling,
        n_negatives_start=n_negatives_start, n_negatives_end=n_negatives_end,
        n_negatives_decay=n_negatives_decay,
        use_cosine_restarts=use_cosine_restarts, restart_period=restart_period,
        cyclic_lr=cyclic_lr, cycle_length=cycle_length, use_c=use_c,
    )

    initial_loss = None
    best_loss = float('inf')
    best_flat_snapshot = None
    last_loss = None
    cancelled = False
    steps_done = 0
    rollbacks_total = 0
    generations = 0

    with Pool(workers, initializer=_init_population,
              initargs=(arch, config, list(ids), worker_kwargs)) as pool:
        while steps_done < steps:
            if cancel_callback is not None and cancel_callback():
                cancelled = True
                break
            k = min(select_every, steps - steps_done)
            tasks = [(list(broadcast_flat), seed + generations * replicas + r, k)
                     for r in range(replicas)]
            results = pool.map(_population_trial, tasks)
            rollbacks_total += sum(r[4] for r in results)
            # Winner = lowest best-loss; ties keep the earlier replica.
            wi = min(range(len(results)),
                     key=lambda i: results[i][0] if _math.isfinite(results[i][0])
                     else float('inf'))
            wbest, wfinal, wflat, wsteps, _wrb = results[wi]
            if initial_loss is None:
                initial_loss = wfinal
            if wbest < best_loss:
                best_loss = wbest
                best_flat_snapshot = list(wflat)
            last_loss = wfinal
            broadcast_flat = list(wflat)
            steps_done += k
            generations += 1
            print(f"generation {generations:3d}  steps {steps_done:5d}/{steps}  "
                  f"winner {wi + 1}/{replicas}  best {wbest:.4f}")
            if progress_cb is not None:
                progress_cb(steps_done, steps, wbest, None, lr)

    if cancelled:
        improved = (initial_loss is not None and best_loss < initial_loss
                    and best_flat_snapshot is not None)
        if improved:
            for prm, val in zip(params, best_flat_snapshot):
                prm.data = val
            if save_on_cancel_callback is not None:
                save_on_cancel_callback(model, best_loss, steps_done)
        else:
            for prm, val in zip(params, initial_params_snapshot):
                prm.data = val
    elif best_flat_snapshot is not None:
        # End on the best weights seen (they may come from any generation).
        for prm, val in zip(params, best_flat_snapshot):
            prm.data = val

    return {
        "cancelled": cancelled,
        "initial_loss": initial_loss,
        "best_loss": best_loss if best_loss != float('inf') else None,
        "final_loss": last_loss,
        "steps_done": steps_done,
        "rollbacks": rollbacks_total,
        "generations": generations,
        "replicas": replicas,
        "kernel_used": "population/Python",
    }
