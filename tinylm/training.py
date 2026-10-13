"""Training: SGD + sampled softmax with cancel support (stdlib only)."""
import math
import random
import time
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Dict, List, Optional

try:
    import accel
except Exception:  # pragma: no cover - accel never raises, belt and braces
    accel = None

from tinylm.autograd import Value
from tinylm.checkpoint import (
    Checkpoint, save_checkpoint, _flat_params,
)
from tinylm.sched import (
    LR_Scheduler, clip_grads, AutoAdjust, RollbackGuard, evaluate, EarlyStopping,
)

# ---- training and generation ----------------------------------------------

def _accel_available():
    """True when the optional C training kernel is loaded (see accel.py)."""
    return accel is not None and accel.AVAILABLE


def _flat_data(model):
    return [p.data for p in model.params]


def _train_c(model, ids, steps, lr, batch_size, n_negatives, progress_cb=None,
             lr_scheduler: str = "linear", warmup_steps: int = 0, min_lr: float = 0.0,
             grad_clip: float = 0.0, val_ids=None, val_split: float = 0.0, eval_every: int = 0,
             early_stopping: bool = False, patience: int = 5, min_delta: float = 1e-4,
             checkpoint_every: int = 0, checkpoint_dir: str = "checkpoints",
             auto_adjust: bool = True, auto_factor: float = 0.5,
             auto_patience: int = 10, auto_clip_relax: bool = True,
             cancel_callback=None, save_on_cancel_callback=None,
             gradient_accumulation: int = 1, neg_sample_seed: int = 0,
             momentum: float = 0.9, rollback_window: int = 0,
             rollback_callback=None):
    """train() via the fused C kernel with gradient clipping support.

    C kernel supports: LR per step, gradient clipping, SGD momentum.
    Python handles: LR scheduling, validation, early stopping, checkpointing,
    automatic adjustment (plateau LR reduction, clip relaxation).
    `cancel_callback()` is checked before each step; on cancel the improved
    weights are kept when loss improved, otherwise initial weights are restored.
    `rollback_window` snapshots every step and restores the lowest-loss step
    in the window when the current loss spikes above it (0 disables).
    `rollback_callback(step, loss, best_loss)` fires on every rollback where
    `best_loss` is the smallest loss seen so far in the run (gap = loss - best).
    Returns a dict with `cancelled`, `initial_loss`, `best_loss`,
    `final_loss`, `steps_done`, `rollbacks`.
    """
    ctx = model.ctx
    vocab = len(model.C)
    emb = len(model.C[0])
    hidden = len(model.b1)
    dims = (vocab, ctx, emb, hidden)
    if len(ids) <= ctx:
        raise ValueError(
            f"Not enough training tokens ({len(ids)}) for ctx={ctx}: "
            "need at least ctx+1 tokens"
        )
    starts = range(len(ids) - ctx)
    flat = _flat_data(model)
    params = model.params

    # Fresh momentum state per run (kernel keeps velocity across steps).
    accel.reset_momentum()

    # Snapshot initial weights so a cancelled run without improvement can be reverted.
    initial_params_snapshot = [p.data for p in params]

    # Setup LR scheduler
    scheduler = LR_Scheduler(lr, steps, lr_scheduler, warmup_steps, min_lr)

    # Setup early stopping
    early_stopper = EarlyStopping(patience=patience, min_delta=min_delta) if early_stopping else None

    # Setup automatic adjustment
    adjuster = AutoAdjust(
        enabled=auto_adjust, factor=auto_factor, patience=auto_patience,
        min_delta=min_delta, warmup_ignore=warmup_steps,
        max_grad_clip=grad_clip,
    )

    # Setup checkpoint directory
    from pathlib import Path
    checkpoint_path = Path(checkpoint_dir)
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    # Split validation data if provided as ratio
    if val_ids is None and val_split > 0:
        split_idx = int(len(ids) * (1 - val_split))
        train_ids = ids[:split_idx]
        val_ids = ids[split_idx:]
        if len(train_ids) <= ctx:
            raise ValueError(
                f"Not enough training tokens after val_split ({len(train_ids)}) "
                f"for ctx={ctx}: need at least ctx+1 tokens"
            )
        starts = range(len(train_ids) - ctx)
    else:
        train_ids = ids

    initial_loss = None
    best_loss = float('inf')
    best_flat_snapshot = None
    last_loss = None
    val_loss = None
    cancelled = False
    steps_done = 0
    rb_guard = RollbackGuard(window=rollback_window, tol=min_delta)

    for step in range(steps):
        # Check for cancellation before starting the next step.
        if cancel_callback is not None and cancel_callback():
            cancelled = True
            steps_done = step
            break
        rate = scheduler.get_lr(step) * adjuster.multiplier
        windows = []
        for _ in range(batch_size):
            i = random.choice(list(starts))
            windows.extend(train_ids[i:i + ctx])
            windows.append(train_ids[i + ctx])

        flat_before = list(flat)
        # C kernel with gradient clipping
        loss, flat = accel.train_step(flat, windows, dims, n_negatives,
                                      rate, random.randrange(1 << 62),
                                      momentum=momentum,
                                      max_norm=adjuster.grad_clip_now())

        # Loss-spike rollback: discard the fused update and jump back to the
        # lowest-loss snapshot in the window. The C velocity already absorbed
        # the spiked grad, so reset it to avoid stepping straight back out.
        if rb_guard.enabled:
            do_rb, best_entry = rb_guard.check(loss)
            if do_rb:
                _, best_flat, _ = best_entry
                flat = list(best_flat)
                for p, x in zip(params, flat):
                    p.data = x
                accel.reset_momentum()
                rb_guard.note_rollback()
                if initial_loss is None:
                    initial_loss = loss
                last_loss = loss
                steps_done = step + 1
                print(f"step {step:5d}  rollback: loss {loss:.3f} > best {best_entry[0]:.3f} in window {rollback_window} -> restored")
                if rollback_callback is not None and best_loss != float('inf'):
                    rollback_callback(step + 1, loss, best_loss)
                if progress_cb is not None:
                    progress_cb(step + 1, steps, loss, val_loss, rate)
                continue
            rb_guard.push(loss, flat_before, None)

        # In-place update of the existing Value objects
        for p, x in zip(params, flat):
            p.data = x

        if initial_loss is None:
            initial_loss = loss
        if loss < best_loss:
            best_loss = loss
            best_flat_snapshot = list(flat)
        last_loss = loss
        steps_done = step + 1

        # Automatic adjustment: reduce LR when loss plateaus
        if adjuster.check(step + 1, loss):
            rate = scheduler.get_lr(step) * adjuster.multiplier
            print(f"step {step:5d}  auto-adjust: lr multiplier -> {adjuster.multiplier:.4f}")

        # Validation evaluation
        val_loss = None
        if val_ids is not None and eval_every > 0 and (step + 1) % eval_every == 0:
            val_loss = evaluate(model, val_ids, batch_size, n_negatives)
            if early_stopper is not None:
                if early_stopper(val_loss, model):
                    print(f"Early stopping at step {step + 1}: val_loss={val_loss:.4f} (best={early_stopper.best_score:.4f})")
                    if progress_cb is not None:
                        progress_cb(step + 1, steps, loss, val_loss, rate)
                    break

        # Checkpointing
        if checkpoint_every > 0 and (step + 1) % checkpoint_every == 0:
            ckpt = Checkpoint(
                step=step + 1,
                model_state=_flat_params(model),
                config=model.get_config(),
                loss=loss,
                val_loss=val_loss,
                lr=rate,
            )
            ckpt_file = checkpoint_path / f"checkpoint_step_{step + 1}.json"
            save_checkpoint(ckpt, str(ckpt_file))

        if progress_cb is not None:
            progress_cb(step + 1, steps, loss, val_loss, rate)

        if step % 25 == 0:
            val_str = f"  val_loss={val_loss:.4f}" if val_loss is not None else ""
            print(f"step {step:5d}  loss {loss:.3f}  lr {rate:.6f}{val_str}")

    # Handle cancellation: keep improved weights, else revert to initial.
    if cancelled:
        improved = (
            initial_loss is not None
            and best_loss < initial_loss
            and best_flat_snapshot is not None
        )
        if improved:
            flat = best_flat_snapshot
            for prm, x in zip(params, flat):
                prm.data = x
            if save_on_cancel_callback is not None:
                save_on_cancel_callback(model, best_loss, steps_done)
        else:
            for prm, x in zip(params, initial_params_snapshot):
                prm.data = x
            flat = _flat_data(model)

    # Save final checkpoint (skip when cancelled — caller decides whether to save)
    if checkpoint_every > 0 and not cancelled and last_loss is not None:
        ckpt = Checkpoint(
            step=steps_done,
            model_state=_flat_params(model),
            config=model.get_config(),
            loss=last_loss,
            val_loss=val_loss,
            lr=scheduler.get_lr(max(0, steps_done - 1)),
        )
        ckpt_file = checkpoint_path / "checkpoint_final.json"
        save_checkpoint(ckpt, str(ckpt_file))

    return {
        "cancelled": cancelled,
        "initial_loss": initial_loss,
        "best_loss": best_loss if best_loss != float('inf') else None,
        "final_loss": last_loss,
        "steps_done": steps_done,
        "rollbacks": rb_guard.rollbacks,
    }


def train(model, ids, steps=2000, lr=0.05, batch_size=16, n_negatives=256,
          progress_cb=None, use_c=None,
          lr_scheduler: str = "cosine_warmup", warmup_steps: int = 100, min_lr: float = 0.0,
          grad_clip: float = 1.0, val_ids=None, val_split: float = 0.0, eval_every: int = 50,
          early_stopping: bool = True, patience: int = 5, min_delta: float = 1e-4,
          checkpoint_every: int = 100, checkpoint_dir: str = "checkpoints",
          auto_adjust: bool = True, auto_factor: float = 0.5,
          auto_patience: int = 10, auto_clip_relax: bool = True,
          gradient_accumulation: int = 1, neg_sample_seed: int = 0,
          momentum: float = 0.9, rollback_window: int = 0,
          rollback_callback=None, cancel_callback=None, save_on_cancel_callback=None):
    """Mini-batch SGD with sampled softmax and advanced training features.

    Each step averages the loss over `batch_size` windows. For each target, the
    softmax runs over the true token plus `n_negatives` random tokens instead of
    the whole vocabulary. This cuts the output-layer cost by roughly
    vocab_size / (n_negatives + 1).

    The optimizer is SGD with momentum: `v = momentum * v + g`, then
    `p -= lr * v`, where `g` is the batch-averaged (optionally gradient-
    clipped) gradient. `momentum=0` falls back to plain SGD. Momentum is what
    makes the loss drop quickly - plain SGD crawls on this architecture.

    New features:
    - SGD with momentum (0 -> plain SGD)
    - LR scheduling: cosine_warmup, linear, constant
    - Gradient clipping by global norm
    - Validation evaluation with perplexity
    - Early stopping with best model restoration
    - Checkpointing for resume capability
    - Automatic adjustment: LR reduction on plateau + auto grad-clip relaxation
    - Gradient accumulation for larger effective batch sizes
    - Pre-computed negative samples for faster training
    - Cancel training with optional save if loss improved
    - Loss-spike rollback: snapshots every step and restores the lowest-loss
      step in `rollback_window` when the current loss spikes above it
      (0 disables; config default 2). `rollback_callback(step, loss, best_loss)`
      fires on every rollback with the run's smallest loss so far.

    `progress_cb(done, total, loss, val_loss=None, lr=None)` is called after every step if given.
    `use_c` forces the optional C kernel on/off; default: on when available.
    Note: C kernel supports LR scheduling and gradient clipping; Python handles validation, early stopping, checkpointing.
    `cancel_callback()` - if provided and returns True, training stops early.
      On cancel: if best loss < initial loss the improved weights are kept
      (and `save_on_cancel_callback(model, best_loss, steps_done)` is called
      when provided); otherwise the model is restored to its initial weights.
    Returns a dict with `cancelled`, `initial_loss`, `best_loss`,
    `final_loss`, `steps_done`, `rollbacks`.
    """
    # Disable C kernel when advanced features are requested that C doesn't support.
    # C kernel now supports: gradient clipping, LR per step (for scheduling).
    # C kernel does NOT natively support: validation, early stopping, checkpointing.
    # Cancel is handled in the C path per-step check as well, but force the
    # Python path when cancellation is requested so save/restore semantics match.
    advanced_features = (
        val_ids is not None or
        val_split > 0 or
        eval_every > 0 or
        early_stopping or
        checkpoint_every > 0 or
        cancel_callback is not None
    )

    if use_c is None:
        use_c = _accel_available() and not advanced_features
    if use_c:
        # C kernel path with gradient clipping support
        return _train_c(model, ids, steps, lr, batch_size, n_negatives, progress_cb,
                 lr_scheduler=lr_scheduler, warmup_steps=warmup_steps, min_lr=min_lr,
                 grad_clip=grad_clip, val_ids=val_ids, val_split=val_split, eval_every=eval_every,
                 early_stopping=early_stopping, patience=patience, min_delta=min_delta,
                 checkpoint_every=checkpoint_every, checkpoint_dir=checkpoint_dir,
                 auto_adjust=auto_adjust, auto_factor=auto_factor,
                 auto_patience=auto_patience, auto_clip_relax=auto_clip_relax,
                 cancel_callback=cancel_callback, save_on_cancel_callback=save_on_cancel_callback,
                 gradient_accumulation=gradient_accumulation, neg_sample_seed=neg_sample_seed,
                 momentum=momentum, rollback_window=rollback_window,
                 rollback_callback=rollback_callback)

    ctx = model.ctx
    if len(ids) <= ctx:
        raise ValueError(
            f"Not enough training tokens ({len(ids)}) for ctx={ctx}: "
            "need at least ctx+1 tokens"
        )
    starts = range(len(ids) - ctx)
    vocab = len(model.C)
    params = model.params

    # Snapshot initial weights so a cancelled run without improvement can be reverted.
    initial_params_snapshot = [p.data for p in params]

    # Pre-compute negative samples for faster training
    if neg_sample_seed:
        neg_rng = random.Random(neg_sample_seed)
    else:
        neg_rng = random

    # Pre-sample negative token candidates for the entire training
    neg_samples = []
    if n_negatives > 0:
        for _ in range(steps * batch_size * n_negatives):
            neg_samples.append(neg_rng.randrange(vocab))
    neg_idx = 0

    # Split validation data if provided as ratio
    if val_ids is None and val_split > 0:
        split_idx = int(len(ids) * (1 - val_split))
        train_ids = ids[:split_idx]
        val_ids = ids[split_idx:]
        if len(train_ids) <= ctx:
            raise ValueError(
                f"Not enough training tokens after val_split ({len(train_ids)}) "
                f"for ctx={ctx}: need at least ctx+1 tokens"
            )
        starts = range(len(train_ids) - ctx)
    else:
        train_ids = ids

    # Setup LR scheduler
    scheduler = LR_Scheduler(lr, steps, lr_scheduler, warmup_steps, min_lr)

    # Setup early stopping
    early_stopper = EarlyStopping(patience=patience, min_delta=min_delta) if early_stopping else None

    # Setup automatic adjustment (LR on plateau + clip relaxation)
    adjuster = AutoAdjust(
        enabled=auto_adjust, factor=auto_factor, patience=auto_patience,
        min_delta=min_delta, warmup_ignore=warmup_steps,
        max_grad_clip=grad_clip, clip_relax=auto_clip_relax,
    )

    # Setup checkpoint directory
    from pathlib import Path
    checkpoint_path = Path(checkpoint_dir)
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    # Track initial/best loss for cancel-and-save-if-improved feature
    initial_loss = None
    best_loss = float('inf')
    best_params_snapshot = None
    last_loss = None
    val_loss = None
    cancelled = False
    steps_done = 0
    rb_guard = RollbackGuard(window=rollback_window, tol=min_delta)

    # Gradient accumulation buffer
    accumulated_grads = [0.0] * len(params)
    accumulation_counter = 0

    # Momentum (SGD) velocity
    velocity = [0.0] * len(params)

    for step in range(steps):
        # Check for cancellation before starting the next step.
        if cancel_callback is not None and cancel_callback():
            cancelled = True
            steps_done = step
            break

        rate = scheduler.get_lr(step) * adjuster.multiplier
        losses = []
        for _ in range(batch_size):
            i = random.choice(starts)
            context, target = train_ids[i:i + ctx], train_ids[i + ctx]

            # Use pre-sampled negatives if available
            if neg_idx + n_negatives <= len(neg_samples):
                negs = neg_samples[neg_idx:neg_idx + n_negatives]
                neg_idx += n_negatives
                losses.append(model.sampled_loss(context, target, n_negatives, vocab, negatives=negs))
            else:
                losses.append(model.sampled_loss(context, target, n_negatives, vocab))

        # Average losses by summing and scaling once
        loss = losses[0]
        for L in losses[1:]:
            loss = loss + L
        loss = loss * (1.0 / batch_size)

        # Loss-spike rollback: if the pre-update loss exceeds the lowest loss
        # in the window, restore that step's snapshot (params + momentum) and
        # skip the bad update. Snapshots are taken every step below.
        if rb_guard.enabled:
            do_rb, best_entry = rb_guard.check(loss.data)
            if do_rb:
                _, best_flat, best_vel = best_entry
                for p, v in zip(params, best_flat):
                    p.data = v
                if best_vel is not None:
                    velocity = list(best_vel)
                accumulated_grads = [0.0] * len(params)
                accumulation_counter = 0
                rb_guard.note_rollback()
                if initial_loss is None:
                    initial_loss = loss.data
                last_loss = loss.data
                steps_done = step + 1
                print(f"step {step:5d}  rollback: loss {loss.data:.3f} > best {best_entry[0]:.3f} in window {rollback_window} -> restored")
                if rollback_callback is not None and best_loss != float('inf'):
                    rollback_callback(step + 1, loss.data, best_loss)
                if progress_cb is not None:
                    progress_cb(step + 1, steps, loss.data, val_loss, rate)
                continue
            rb_guard.push(loss.data, [p.data for p in params], velocity)

        # Track initial and best loss (pre-update loss)
        if initial_loss is None:
            initial_loss = loss.data
        if loss.data < best_loss:
            best_loss = loss.data
            best_params_snapshot = [p.data for p in params]
        last_loss = loss.data

        for p in params:
            p.grad = 0.0
        loss.backward()

        # Gradient clipping (with automatic relaxation if chronically clipping)
        clip_now = adjuster.grad_clip_now()
        if clip_now > 0:
            norm = clip_grads(params, clip_now)
            adjuster.note_grad_clip(norm)

        # Gradient accumulation
        accumulation_counter += 1
        for i, p in enumerate(params):
            accumulated_grads[i] += p.grad

        if accumulation_counter >= gradient_accumulation:
            # Apply accumulated gradients with SGD momentum:
            # v = momentum * v + g ; p -= lr * v (momentum == 0 -> plain SGD)
            # `p.grad` is the gradient of the batch-AVERAGED loss, so only
            # average across accumulation steps - dividing by batch_size again
            # would silently shrink the effective LR by batch_size.
            scale = 1.0 / gradient_accumulation
            for i, p in enumerate(params):
                velocity[i] = momentum * velocity[i] + accumulated_grads[i] * scale
                p.data -= rate * velocity[i]
                accumulated_grads[i] = 0.0
            accumulation_counter = 0

        # Automatic adjustment: reduce LR when loss plateaus
        if adjuster.check(step + 1, loss.data):
            rate = scheduler.get_lr(step) * adjuster.multiplier
            print(f"step {step:5d}  auto-adjust: lr multiplier -> {adjuster.multiplier:.4f}")

        # Validation evaluation
        val_loss = None
        if val_ids is not None and eval_every > 0 and (step + 1) % eval_every == 0:
            val_loss = evaluate(model, val_ids, batch_size, n_negatives)
            # Early stopping check
            if early_stopper is not None:
                if early_stopper(val_loss, model):
                    print(f"Early stopping at step {step + 1}: val_loss={val_loss:.4f} (best={early_stopper.best_score:.4f})")
                    if progress_cb is not None:
                        progress_cb(step + 1, steps, loss.data, val_loss, rate)
                    steps_done = step + 1
                    break

        # Checkpointing
        if checkpoint_every > 0 and (step + 1) % checkpoint_every == 0:
            ckpt = Checkpoint(
                step=step + 1,
                model_state=_flat_params(model),
                config=model.get_config(),
                loss=loss.data,
                val_loss=val_loss,
                lr=rate,
            )
            ckpt_file = checkpoint_path / f"checkpoint_step_{step + 1}.json"
            save_checkpoint(ckpt, str(ckpt_file))

        if progress_cb is not None:
            progress_cb(step + 1, steps, loss.data, val_loss, rate)

        if step % 25 == 0:
            val_str = f"  val_loss={val_loss:.4f}" if val_loss is not None else ""
            print(f"step {step:5d}  loss {loss.data:.3f}  lr {rate:.6f}{val_str}")

        steps_done = step + 1

    # Handle cancellation: keep improved weights, else revert to initial.
    if cancelled:
        improved = (
            initial_loss is not None
            and best_loss < initial_loss
            and best_params_snapshot is not None
        )
        if improved:
            for p, val in zip(params, best_params_snapshot):
                p.data = val
            if save_on_cancel_callback is not None:
                save_on_cancel_callback(model, best_loss, steps_done)
        else:
            for p, val in zip(params, initial_params_snapshot):
                p.data = val

    # Save final checkpoint (skip when cancelled — caller decides whether to save)
    if checkpoint_every > 0 and not cancelled:
        ckpt = Checkpoint(
            step=steps_done,
            model_state=_flat_params(model),
            config=model.get_config(),
            loss=last_loss,
            val_loss=val_loss,
            lr=scheduler.get_lr(max(0, steps_done - 1)),
        )
        ckpt_file = checkpoint_path / "checkpoint_final.json"
        save_checkpoint(ckpt, str(ckpt_file))

    return {
        "cancelled": cancelled,
        "initial_loss": initial_loss,
        "best_loss": best_loss if best_loss != float('inf') else None,
        "final_loss": last_loss,
        "steps_done": steps_done,
        "rollbacks": rb_guard.rollbacks,
    }
