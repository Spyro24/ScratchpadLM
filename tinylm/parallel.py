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
from tinylm.sched import RollbackGuard, clip_grads

class _WorkerState:
    """Holds a model instance in each worker process to avoid reconstruction."""
    model = None
    cfg = None
    vocab_size = 0
    n_negatives = 0
    use_c = False
    max_norm = 0.0


def _init_worker(flat, cfg, vocab_size, n_negatives, max_norm=0.0):
    """Initializer for persistent worker pool."""
    from tinylm.models_tiny import TinyLM
    from tinylm.training import _accel_available
    _WorkerState.cfg = cfg
    _WorkerState.vocab_size = vocab_size
    _WorkerState.n_negatives = n_negatives
    _WorkerState.max_norm = max_norm
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
    """Worker function that receives (flat, windows) and uses persistent model."""
    flat, windows = args
    if _WorkerState.use_c:
        cfg = _WorkerState.cfg
        dims = (cfg["vocab"], cfg["ctx"], cfg["emb"], cfg["hidden"])
        flat_windows = []
        for context, target in windows:
            flat_windows.extend(context)
            flat_windows.append(target)
        return accel.batch_loss_grads(flat, flat_windows, dims,
                                      _WorkerState.n_negatives,
                                      random.randrange(1 << 62),
                                      max_norm=_WorkerState.max_norm)
    model = _WorkerState.model
    _set_flat_params(model, flat)

    losses = []
    for context, target in windows:
        losses.append(model.sampled_loss(context, target, _WorkerState.n_negatives, _WorkerState.vocab_size))
    # Sum losses
    loss = losses[0]
    for L in losses[1:]:
        loss = loss + L
    loss.backward()
    # Apply gradient clipping in Python path
    if _WorkerState.max_norm > 0:
        clip_grads(model.params, _WorkerState.max_norm)
    return loss.data, [p.grad for p in model.params]


def train_parallel(model, ids, steps=2000, lr=0.05, batch_size=16, n_negatives=256, workers=None,
                   progress_cb=None, grad_clip: float = 0.0, cancel_callback=None,
                   save_on_cancel_callback=None, momentum: float = 0.9,
                   rollback_window: int = 0, rollback_tol: float = 1e-4,
                   rollback_callback=None):
    """Data-parallel version of train. Uses persistent worker pool to avoid
    rebuilding the model each step. The batch is split across processes and
    the gradients are summed before one SGD update (with momentum).

    Matches `train()`: gradients are averaged over the batch, clipped once by
    global norm, then applied as `v = momentum * v + g ; p -= lr * v`
    (`momentum=0` -> plain SGD).

    `rollback_window` snapshots every step and restores the lowest-loss step
    in the window when the current loss spikes above it by more than
    `rollback_tol` (0 disables). `rollback_callback(step, loss, best_loss)`
    fires on every rollback with the run's smallest loss so far.

    `progress_cb(done, total, loss)` is called after every step if given
    (runs in the main process; safe for queue-based UI updates).
    `cancel_callback()` - if provided and returns True, training stops early.
      On cancel: if best loss < initial loss the improved weights are kept
      (and `save_on_cancel_callback(model, best_loss, steps_done)` is called
      when provided); otherwise the model is restored to its initial weights.
    Returns a dict with `cancelled`, `initial_loss`, `best_loss`,
    `final_loss`, `steps_done`, `rollbacks`.
    """
    workers = workers or cpu_count()
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

    # Pre-create initial flat params for worker initialization.
    # Workers only compute gradients: clipping is applied once in the parent
    # on the batch-averaged gradient, so the in-worker clip stays off.
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
            batch = [(ids[i:i + ctx], ids[i + ctx])
                     for i in (random.choice(starts) for _ in range(batch_size))]
            chunks = [batch[k::workers] for k in range(workers)]
            chunks = [c for c in chunks if c]

            flat = _flat_params(model)
            # Send (flat, chunk) to each worker
            results = pool.map(_worker_grads_v2, [(flat, c) for c in chunks])

            total_loss = sum(r[0] for r in results)
            grad = [0.0] * len(params)
            for _, g in results:
                for k, x in enumerate(g):
                    grad[k] += x
            # Average the summed gradients over the batch (one update per step)
            grad = [g / batch_size for g in grad]

            # Track initial and best loss (pre-update loss)
            step_loss = total_loss / batch_size
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
            if initial_loss is None:
                initial_loss = step_loss
            if step_loss < best_loss:
                best_loss = step_loss
                best_params_snapshot = [p.data for p in params]
            last_loss = step_loss
            steps_done = step + 1

            # Gradient clipping on the mean gradient (same rule as train())
            if grad_clip > 0:
                import math
                total_norm = math.sqrt(sum(g * g for g in grad))
                if total_norm > grad_clip and total_norm > 0:
                    scale = grad_clip / total_norm
                    grad = [g * scale for g in grad]

            # SGD with momentum: v = momentum * v + g ; p -= rate * v
            for i, p in enumerate(params):
                velocity[i] = momentum * velocity[i] + grad[i]
                p.data -= rate * velocity[i]

            if progress_cb is not None:
                progress_cb(step + 1, steps, step_loss)

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
    }
