"""Training helpers: LR schedule, grad clipping, auto-adjust, eval, early stopping."""
import math
import random
from typing import List, Optional

from tinylm.autograd import Value

# ---- LR Scheduler -----------------------------------------------------------

class LR_Scheduler:
    """Learning rate scheduler with multiple strategies."""

    def __init__(self, base_lr: int, total_steps: int, scheduler: str = "cosine_warmup",
                 warmup_steps: int = 0, min_lr: float = 0.0):
        self.base_lr = base_lr
        self.total_steps = total_steps
        self.scheduler = scheduler
        self.warmup_steps = min(warmup_steps, total_steps)
        self.min_lr = min_lr

    def get_lr(self, step: int) -> float:
        """Get learning rate for given step (0-indexed)."""
        if step < self.warmup_steps:
            # Linear warmup
            return self.base_lr * (step + 1) / max(1, self.warmup_steps)

        progress = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
        progress = min(1.0, max(0.0, progress))

        if self.scheduler == "cosine_warmup":
            # Cosine decay to min_lr
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            return self.min_lr + (self.base_lr - self.min_lr) * cosine_decay
        elif self.scheduler == "linear":
            # Linear decay to min_lr
            return self.min_lr + (self.base_lr - self.min_lr) * (1.0 - progress)
        else:  # constant
            return self.base_lr


# ---- Gradient Clipping ------------------------------------------------------

def clip_grads(params, max_norm: float = 1.0) -> float:
    """Clip gradients by global norm. Returns the global norm before clipping."""
    total_norm = 0.0
    for p in params:
        total_norm += p.grad * p.grad
    total_norm = math.sqrt(total_norm)

    if total_norm > max_norm and total_norm > 0:
        clip_coef = max_norm / (total_norm + 1e-6)
        for p in params:
            p.grad *= clip_coef

    return total_norm


# ---- Automatic adjustment ---------------------------------------------------

class AutoAdjust:
    """Automatic learning-rate adjustment on plateaus (ReduceLROnPlateau).

    Tracks the running loss; when no improvement of at least `min_delta` is
    seen for `patience` consecutive checks, the LR multiplier is scaled by
    `factor` (floored at `floor`). Works with any LR schedule: the returned
    multiplier is applied on top of `LR_Scheduler.get_lr(step)`.

    Also exposes `grad_clip_now()` so the effective gradient-clip threshold
    can be relaxed if clipping is firing on every step (clip too tight).
    """

    def __init__(self, enabled: bool = True, factor: float = 0.5,
                 patience: int = 10, min_delta: float = 1e-3,
                 floor: float = 1e-4, warmup_ignore: int = 0,
                 check_every: int = 1, clip_relax_after: int = 5,
                 clip_relax_factor: float = 2.0, max_grad_clip: float = 0.0):
        self.enabled = enabled
        self.factor = factor
        self.patience = patience
        self.min_delta = min_delta
        self.floor = floor
        self.warmup_ignore = warmup_ignore   # ignore first N steps (warmup)
        self.check_every = check_every
        self.clip_relax_after = clip_relax_after   # consecutive clipped steps
        self.clip_relax_factor = clip_relax_factor
        self.max_grad_clip = max_grad_clip   # 0 = no auto clip relaxation

        self.multiplier = 1.0
        self.best_loss = None
        self.bad_checks = 0
        self.last_check_step = -1
        self.reduce_count = 0
        self._clip_streak = 0
        self._clip_limit = max_grad_clip

    def check(self, step: int, loss: float) -> bool:
        """Update state with a new loss; return True if LR was reduced.

        Best loss is tracked from step 1 (even during warmup), but no
        reduction ever happens at or before `warmup_ignore` steps.
        """
        if not self.enabled:
            return False
        improved = self.best_loss is None or loss < self.best_loss - self.min_delta
        if improved:
            self.best_loss = loss
            self.bad_checks = 0
            return False
        if step <= self.warmup_ignore:
            return False
        if self.check_every > 1 and step % self.check_every != 0:
            return False
        self.bad_checks += 1
        if self.bad_checks >= self.patience:
            new_mult = max(self.floor, self.multiplier * self.factor)
            if new_mult < self.multiplier:
                self.multiplier = new_mult
                self.bad_checks = 0
                self.reduce_count += 1
                return True
            self.bad_checks = 0
        return False

    def note_grad_clip(self, norm: float) -> None:
        """Track whether gradient clipping is firing, relax limit if chronic."""
        if not self.enabled or self.max_grad_clip <= 0:
            return
        if norm > self._clip_limit:
            self._clip_streak += 1
            if self._clip_streak >= self.clip_relax_after:
                self._clip_limit *= self.clip_relax_factor
                self._clip_streak = 0
        else:
            self._clip_streak = 0

    def grad_clip_now(self) -> float:
        """Effective gradient-clip threshold (0 disables clipping)."""
        return self._clip_limit


# ---- Evaluation -------------------------------------------------------------

def evaluate(model, val_ids, batch_size=16, n_negatives=256) -> float:
    """Evaluate model on validation data. Returns perplexity."""
    ctx = model.ctx
    vocab = len(model.C)
    starts = range(len(val_ids) - ctx)
    params = model.params

    total_loss = 0.0
    n_batches = min(batch_size, len(starts))
    if n_batches == 0:
        return float('inf')

    for _ in range(n_batches):
        i = random.choice(starts)
        context, target = val_ids[i:i + ctx], val_ids[i + ctx]
        loss = model.sampled_loss(context, target, n_negatives, vocab)
        total_loss += loss.data

        for p in params:
            p.grad = 0.0
        loss.backward()

    avg_loss = total_loss / n_batches
    perplexity = math.exp(min(avg_loss, 20))  # Cap to avoid overflow
    return perplexity


# ---- Early Stopping ---------------------------------------------------------

class EarlyStopping:
    """Early stopping with best model restoration."""

    def __init__(self, patience: int = 5, min_delta: float = 1e-4, mode: str = "min"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode  # "min" for loss/perplexity, "max" for accuracy
        self.best_score = None
        self.counter = 0
        self.best_params = None
        self.should_stop = False

    def __call__(self, score: float, model) -> bool:
        """Check if training should stop. Returns True if should stop."""
        if self.mode == "min":
            improved = self.best_score is None or score < self.best_score - self.min_delta
        else:
            improved = self.best_score is None or score > self.best_score + self.min_delta

        if improved:
            self.best_score = score
            self.counter = 0
            # Save best model parameters
            self.best_params = [p.data for p in model.params]
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                # Restore best parameters
                if self.best_params is not None:
                    for p, val in zip(model.params, self.best_params):
                        p.data = val
                return True
        return False

    def restore_best(self, model):
        """Manually restore best parameters."""
        if self.best_params is not None:
            for p, val in zip(model.params, self.best_params):
                p.data = val


# ---- Checkpointing ----------------------------------------------------------
