"""Training helpers: LR schedule, grad clipping, auto-adjust, eval, early stopping, APOLLO optimizer."""
import math
import random
from collections import deque
from typing import List, Optional, Tuple

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
    """Automatic learning-rate adjustment on plateaus (ReduceLROnPlateau) with
    cosine annealing restarts and adaptive momentum.

    Features:
    - ReduceLROnPlateau with configurable patience, factor, floor
    - Cosine annealing with warm restarts (SGDR-style) when enabled
    - Adaptive momentum decay on plateaus
    - Gradient clip relaxation when chronic clipping
    - LR warmup with configurable schedule
    - Optional EMA smoothing for more stable plateau detection
    - Minimum LR enforcement with optional cyclic learning rates
    """

    def __init__(self, enabled: bool = True, factor: float = 0.5,
                 patience: int = 10, min_delta: float = 1e-3,
                 floor: float = 1e-4, warmup_ignore: int = 0,
                 check_every: int = 1, clip_relax_after: int = 5,
                 clip_relax_factor: float = 2.0, max_grad_clip: float = 0.0,
                 clip_relax: bool = True, momentum: float = 0.9,
                 # New features:
                 use_cosine_restarts: bool = False, restart_period: int = 0,
                 restart_mult: float = 1.0, ema_alpha: float = 0.9,
                 use_ema: bool = False,  # EMA for plateau detection (off by default for backward compat)
                 min_lr: float = 1e-6, cyclic_lr: bool = False,
                 cycle_length: int = 0):
        self.enabled = enabled
        self.factor = factor
        self.patience = patience
        self.min_delta = min_delta
        self.floor = floor
        self.warmup_ignore = warmup_ignore
        self.check_every = check_every
        self.clip_relax_after = clip_relax_after
        self.clip_relax_factor = clip_relax_factor
        self.max_grad_clip = max_grad_clip
        self.clip_relax = clip_relax
        self.momentum = momentum

        # New features
        self.use_cosine_restarts = use_cosine_restarts
        self.restart_period = restart_period
        self.restart_mult = restart_mult
        self.ema_alpha = ema_alpha
        self.use_ema = use_ema
        self.min_lr = min_lr
        self.cyclic_lr = cyclic_lr
        self.cycle_length = cycle_length

        self.multiplier = 1.0
        self.best_loss = None
        self.ema_loss = None
        self.bad_checks = 0
        self.reduce_count = 0
        self._clip_streak = 0
        self._clip_limit = max_grad_clip
        self._restart_step = 0
        self._cycle_step = 0

    def check(self, step: int, loss: float) -> bool:
        """Update state with a new loss; return True if LR was reduced.

        Uses EMA-smoothed loss for plateau detection if enabled.
        Supports cosine restarts and cyclic LR.
        """
        if not self.enabled:
            return False

        # Update EMA loss if enabled
        if self.use_ema:
            if self.ema_loss is None:
                self.ema_loss = loss
            else:
                self.ema_loss = self.ema_alpha * self.ema_loss + (1 - self.ema_alpha) * loss

        # Use EMA loss for plateau detection if enabled, else raw loss
        current_loss = self.ema_loss if self.use_ema else loss
        improved = self.best_loss is None or current_loss < self.best_loss - self.min_delta

        if improved:
            self.best_loss = current_loss
            self.bad_checks = 0
        elif step > self.warmup_ignore:
            if self.check_every == 1 or step % self.check_every == 0:
                self.bad_checks += 1

        # Cosine annealing with warm restarts
        if self.use_cosine_restarts and self.restart_period > 0:
            cycle_pos = (step - self._restart_step) % self.restart_period
            if cycle_pos == 0 and step > self._restart_step:
                # Restart: briefly increase LR
                self.multiplier = min(1.0, self.multiplier * self.restart_mult)
                self._restart_step = step

        # Cyclic LR
        if self.cyclic_lr and self.cycle_length > 0:
            self._cycle_step = step % self.cycle_length
            cycle_progress = self._cycle_step / self.cycle_length
            # Triangle wave: up then down
            if cycle_progress < 0.5:
                cyclic_mult = 1.0 + (self.multiplier - 1.0) * (2 * cycle_progress)
            else:
                cyclic_mult = 1.0 + (self.multiplier - 1.0) * (2 - 2 * cycle_progress)
            self.multiplier = max(self.min_lr / self.best_loss if self.best_loss else self.min_lr, cyclic_mult)

        # Reduce on plateau
        reduced = False
        if self.bad_checks >= self.patience:
            new_mult = max(self.floor, self.multiplier * self.factor)
            if new_mult < self.multiplier:
                self.multiplier = new_mult
                # Decay momentum on plateau
                self.momentum = max(0.0, self.momentum * self.factor)
                self.bad_checks = 0
                self.reduce_count += 1
                reduced = True
            else:
                self.bad_checks = 0

        return reduced

    def note_grad_clip(self, norm: float) -> None:
        """Track whether gradient clipping is firing, relax limit if chronic."""
        if not self.enabled or not self.clip_relax or self.max_grad_clip <= 0:
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

    def get_cosine_restart_lr(self, base_lr: float, step: int) -> float:
        """Get LR with cosine annealing restarts applied."""
        if not self.use_cosine_restarts or self.restart_period <= 0:
            return base_lr * self.multiplier
        cycle_pos = (step - self._restart_step) % self.restart_period
        progress = cycle_pos / self.restart_period
        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
        return self.min_lr + (base_lr * self.multiplier - self.min_lr) * cosine_decay


# ---- Loss-spike rollback ----------------------------------------------------

class RollbackGuard:
    """Per-step snapshot + spike rollback to the best step in a window.

    Every training step snapshots (loss, flat params, velocity). When the
    window holds `window` entries and the current pre-update loss exceeds the
    lowest loss in the window by more than `max(tol, rel_tol * |best|)`,
    the caller restores the snapshot of that lowest-loss step ("jumps back")
    and skips the bad update. The relative term keeps normal batch noise
    (losses here are ~1-10, so absolute `tol` alone would fire constantly)
    from triggering rollbacks; only real spikes do.

    `window=0` disables rollback. Recommended window is 2. After a rollback
    the window is collapsed to the best entry and a cooldown of `window`
    steps starts during which finite-loss spikes are ignored (the restored
    weights need a few steps to establish a fresh baseline instead of
    pin-looping against one lucky-low batch). Non-finite losses always roll
    back, cooldown or not.
    """

    def __init__(self, window: int = 2, tol: float = 0.0, rel_tol: float = 0.10):
        self.window = int(window)
        self.tol = float(tol)
        self.rel_tol = float(rel_tol)
        self.history: Optional[deque] = (
            deque(maxlen=self.window) if self.window > 0 else None
        )
        self.rollbacks = 0
        self._cooldown = 0

    @property
    def enabled(self) -> bool:
        return self.window > 0 and self.history is not None

    def check(self, loss: float) -> Tuple[bool, Optional[tuple]]:
        """Return (should_rollback, best_entry). Best entry is
        (loss, flat, velocity-or-None). Non-finite loss always rolls back
        (to the best finite snapshot). Finite loss rolls back only once the
        window is full, the post-rollback cooldown has expired, and the loss
        exceeds the window best by `max(tol, rel_tol * |best|)`.
        """
        if not self.enabled or not self.history:
            return False, None
        if not math.isfinite(loss):
            finite = [e for e in self.history if math.isfinite(e[0])]
            if not finite:
                return False, None
            return True, min(finite, key=lambda e: e[0])
        if self._cooldown > 0:
            return False, None
        if len(self.history) < self.window:
            return False, None
        best = min(self.history, key=lambda e: e[0])
        limit = best[0] + max(self.tol, self.rel_tol * abs(best[0]))
        if loss > limit:
            return True, best
        return False, None

    def push(self, loss: float, flat: List[float], velocity=None) -> None:
        """Snapshot the pre-update params/velocity that produced `loss`."""
        if not self.enabled:
            return
        self.history.append(
            (float(loss), list(flat),
             list(velocity) if velocity is not None else None)
        )
        if self._cooldown > 0:
            self._cooldown -= 1

    def note_rollback(self) -> None:
        """Count a rollback, collapse the window to its best entry, and start
        a `window`-step cooldown so the restored weights are not immediately
        rolled back again on normal batch noise."""
        self.rollbacks += 1
        if self.enabled and self.history:
            best = min(self.history, key=lambda e: e[0])
            self.history.clear()
            self.history.append(best)
            self._cooldown = self.window


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


# ---- APOLLO Optimizer ---------------------------------------------------------

class APOLLOOptimizer:
    """APOLLO: SGD-like Memory, AdamW-level Performance.
    
    Implements structured channel-wise gradient scaling using low-rank
    random projections. Achieves AdamW-level performance with SGD-like memory.
    
    Args:
        params: List of model parameters (Value objects)
        lr: Base learning rate
        rank: Rank of the random projection (default: min(256, param_count // 4))
        beta1: First moment decay
        beta2: Second moment decay
        eps: Numerical stability constant
        weight_decay: Decoupled weight decay
        proj_update_freq: Frequency to resample projection matrix
        norm_growth_limit: Norm-growth limiter threshold (gamma), 0 to disable
        min_lr: Minimum learning rate floor
    """
    
    def __init__(self, params: List[Value], lr: float = 0.01, rank: int = 256,
                 beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8,
                 weight_decay: float = 0.01, proj_update_freq: int = 200,
                 norm_growth_limit: float = 1.01, min_lr: float = 1e-6):
        self.params = params
        self.lr = lr
        self.rank = rank
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.proj_update_freq = proj_update_freq
        self.norm_growth_limit = norm_growth_limit
        self.min_lr = min_lr
        
        self.step_count = 0
        self.param_shapes = []
        self.param_slices = []
        self._init_param_layout()
        
        # Low-rank optimizer states
        self.M_low = None  # First moment in low-rank space
        self.V_low = None  # Second moment in low-rank space
        self.P = None      # Random projection matrix
        self.prev_grad_norm = None
        
    def _init_param_layout(self):
        """Initialize parameter layout for projection."""
        total_params = len(self.params)
        self.rank = min(self.rank, max(1, total_params // 4))
        
        # Flatten all params into a single vector for projection
        # Track shapes for reconstruction
        offset = 0
        for p in self.params:
            self.param_shapes.append((1,))  # Each param is scalar in our case
            self.param_slices.append(slice(offset, offset + 1))
            offset += 1
            
    def _get_flat_grad(self) -> List[float]:
        """Get flattened gradients."""
        return [p.grad for p in self.params]
    
    def _get_flat_data(self) -> List[float]:
        """Get flattened parameter data."""
        return [p.data for p in self.params]
    
    def _set_flat_data(self, flat_data: List[float]):
        """Set parameter data from flat array."""
        for p, val in zip(self.params, flat_data):
            p.data = val
    
    def _init_projection(self, grad_dim: int):
        """Initialize or resample random projection matrix P ~ N(0, 1/rank)."""
        import random
        rng = random.Random(self.step_count)  # Deterministic seed based on step
        self.P = [[rng.gauss(0.0, 1.0 / math.sqrt(self.rank)) 
                   for _ in range(grad_dim)] for _ in range(self.rank)]
        # Initialize low-rank moments
        self.M_low = [[0.0 for _ in range(grad_dim)] for _ in range(self.rank)]
        self.V_low = [[0.0 for _ in range(grad_dim)] for _ in range(self.rank)]
    
    def _project_grad(self, grad: List[float]) -> List[List[float]]:
        """Project gradient to low-rank space: R = P @ G."""
        # grad is 1D (flattened params), treat as column vector
        # R[rank][1] = P[rank][grad_dim] @ G[grad_dim][1]
        R = [[0.0] for _ in range(self.rank)]
        for i in range(self.rank):
            s = 0.0
            Pi = self.P[i]
            for j, g in enumerate(grad):
                s += Pi[j] * g
            R[i][0] = s
        return R
    
    def _update_low_rank_moments(self, R: List[List[float]]):
        """Update low-rank first and second moments (AdamW style)."""
        for i in range(self.rank):
            r_val = R[i][0]
            # First moment
            self.M_low[i][0] = self.beta1 * self.M_low[i][0] + (1 - self.beta1) * r_val
            # Second moment
            self.V_low[i][0] = self.beta2 * self.V_low[i][0] + (1 - self.beta2) * (r_val * r_val)
    
    def _compute_scaling_factors(self) -> List[float]:
        """Compute channel-wise scaling factors s_j = ||R~_j|| / ||R_j||.
        
        For APOLLO (channel-wise): each output dimension gets its own scaling.
        Since we have rank x 1 projection, we compute tensor-wise scaling.
        """
        # R_tilde = M_low / sqrt(V_low + eps)
        R_tilde = [[0.0] for _ in range(self.rank)]
        for i in range(self.rank):
            denom = math.sqrt(self.V_low[i][0] + self.eps)
            R_tilde[i][0] = self.M_low[i][0] / denom if denom > 0 else 0.0
        
        # Tensor-wise scaling (APOLLO-Mini style): single scalar for all params
        norm_R_tilde = math.sqrt(sum(r[0] * r[0] for r in R_tilde))
        norm_R = math.sqrt(sum(r[0] * r[0] for r in self.M_low))  # Using M_low as proxy for R
        
        if norm_R > 0 and norm_R_tilde > 0:
            scale = norm_R_tilde / norm_R
        else:
            scale = 1.0
        
        # Return per-parameter scaling (all same for tensor-wise)
        return [scale] * len(self.params)
    
    def _apply_norm_growth_limiter(self, scaled_grad: List[float]) -> List[float]:
        """Apply norm-growth limiter: if ||g_t||/||g_{t-1}|| > gamma, scale down."""
        if self.norm_growth_limit <= 0:
            return scaled_grad
        
        grad_norm = math.sqrt(sum(g * g for g in scaled_grad))
        if self.prev_grad_norm is not None and self.prev_grad_norm > 0:
            ratio = grad_norm / self.prev_grad_norm
            if ratio > self.norm_growth_limit:
                # Scale down
                factor = self.norm_growth_limit / ratio
                scaled_grad = [g * factor for g in scaled_grad]
                grad_norm *= factor
        self.prev_grad_norm = grad_norm
        return scaled_grad
    
    def step(self):
        """Perform one optimization step."""
        self.step_count += 1
        
        # Get flat gradients
        flat_grad = self._get_flat_grad()
        grad_dim = len(flat_grad)
        
        # Initialize projection on first step or resample periodically
        if self.P is None or (self.proj_update_freq > 0 and self.step_count % self.proj_update_freq == 0):
            self._init_projection(grad_dim)
        
        # Project gradient to low-rank space
        R = self._project_grad(flat_grad)
        
        # Update low-rank moments
        self._update_low_rank_moments(R)
        
        # Compute scaling factors
        scales = self._compute_scaling_factors()
        
        # Apply scaling to gradients: G_tilde = G * S
        scaled_grad = [g * s for g, s in zip(flat_grad, scales)]
        
        # Apply norm-growth limiter
        scaled_grad = self._apply_norm_growth_limiter(scaled_grad)
        
        # Apply weight decay (decoupled)
        if self.weight_decay > 0:
            flat_data = self._get_flat_data()
            for i in range(len(scaled_grad)):
                scaled_grad[i] += self.weight_decay * flat_data[i]
        
        # Update parameters: p -= lr * scaled_grad
        flat_data = self._get_flat_data()
        for i in range(len(flat_data)):
            flat_data[i] -= self.lr * scaled_grad[i]
        
        self._set_flat_data(flat_data)
        
        # Zero gradients
        for p in self.params:
            p.grad = 0.0


class APOLLOMiniOptimizer(APOLLOOptimizer):
    """APOLLO-Mini: Extreme memory efficiency with rank-1 tensor-wise scaling.
    
    Uses only a single scalar for gradient scaling (rank=1), achieving
    SGD-level memory while outperforming AdamW.
    """
    
    def __init__(self, params: List[Value], lr: float = 0.01, **kwargs):
        # Force rank=1 for APOLLO-Mini (pop from kwargs if provided)
        kwargs.pop('rank', None)
        super().__init__(params, lr, rank=1, **kwargs)
    
    def _compute_scaling_factors(self) -> List[float]:
        """Compute tensor-wise scaling factor (single scalar for all params)."""
        # R_tilde = M_low / sqrt(V_low + eps)
        R_tilde_val = 0.0
        R_val = 0.0
        for i in range(self.rank):
            denom = math.sqrt(self.V_low[i][0] + self.eps)
            r_tilde = self.M_low[i][0] / denom if denom > 0 else 0.0
            R_tilde_val += r_tilde * r_tilde
            R_val += self.M_low[i][0] * self.M_low[i][0]
        
        norm_R_tilde = math.sqrt(R_tilde_val)
        norm_R = math.sqrt(R_val)
        
        if norm_R > 0 and norm_R_tilde > 0:
            scale = norm_R_tilde / norm_R
        else:
            scale = 1.0
        
        # Return per-parameter scaling (all same for tensor-wise)
        return [scale] * len(self.params)


# ---- Adam / Lion / Muon optimizers -------------------------------------------

class AdamOptimizer:
    """AdamW: per-parameter adaptive LR with decoupled weight decay.

    Standard Adam (Kingma & Ba) with bias correction + decoupled weight
    decay (Loshchilov & Hutter). Still the most reliable loss reducer for
    small models; O(params) time and 2 state floats per param.

    Contract matches APOLLOOptimizer: `lr` is refreshed from the scheduler
    each step, `step()` reads `p.grad`, updates `p.data` in place and
    zeroes grads. `lr_scale` maps the shared base LR into Adam range
    (e.g. base 0.1 * 0.03 = 3e-3).
    """

    def __init__(self, params: List[Value], lr: float = 0.01,
                 betas: Tuple[float, float] = (0.9, 0.999), eps: float = 1e-8,
                 weight_decay: float = 0.0, lr_scale: float = 1.0):
        self.params = params
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.lr_scale = lr_scale
        self.m = [0.0] * len(params)
        self.v = [0.0] * len(params)
        self.t = 0

    def step(self, indices=None):
        """One AdamW step over `indices` (default: all params)."""
        self.t += 1
        lr_eff = self.lr * self.lr_scale
        b1, b2 = self.beta1, self.beta2
        bc1 = 1.0 - b1 ** self.t
        bc2 = 1.0 - b2 ** self.t
        if indices is None:
            indices = range(len(self.params))
        for i in indices:
            p = self.params[i]
            g = p.grad
            m = b1 * self.m[i] + (1.0 - b1) * g
            v = b2 * self.v[i] + (1.0 - b2) * g * g
            self.m[i], self.v[i] = m, v
            denom = math.sqrt(v / bc2) + self.eps
            p.data -= lr_eff * (m / bc1 / denom + self.weight_decay * p.data)
            p.grad = 0.0


class LionOptimizer:
    """Lion: sign-momentum optimizer (Chen et al., 2023), AdamW-level loss.

    Update: `m = b1*m + (1-b1)*g` for direction, `p -= lr*(sign(m) + wd*p)`,
    momentum carried with `b2`. O(params) time, 1 state float per param.
    Lion wants a smaller LR than AdamW (~1/10); `lr_scale` maps the shared
    base LR (e.g. base 0.1 * 0.003 = 3e-4). Same `step()` contract as Adam.
    """

    def __init__(self, params: List[Value], lr: float = 0.01,
                 betas: Tuple[float, float] = (0.9, 0.99),
                 weight_decay: float = 0.0, lr_scale: float = 1.0):
        self.params = params
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.weight_decay = weight_decay
        self.lr_scale = lr_scale
        self.m = [0.0] * len(params)

    def step(self, indices=None):
        """One Lion step over `indices` (default: all params)."""
        lr_eff = self.lr * self.lr_scale
        b1, b2 = self.beta1, self.beta2
        if indices is None:
            indices = range(len(self.params))
        for i in indices:
            p = self.params[i]
            g = p.grad
            c = b1 * self.m[i] + (1.0 - b1) * g
            s = 1.0 if c > 0.0 else (-1.0 if c < 0.0 else 0.0)
            p.data -= lr_eff * (s + self.weight_decay * p.data)
            self.m[i] = b2 * self.m[i] + (1.0 - b2) * g
            p.grad = 0.0


def _matmul(A: List[List[float]], B: List[List[float]]) -> List[List[float]]:
    """Plain matmul (stdlib only). A: m×k, B: k×n -> m×n."""
    m, k = len(A), len(A[0])
    n = len(B[0])
    out = [[0.0] * n for _ in range(m)]
    for i in range(m):
        Ai, oi = A[i], out[i]
        for p in range(k):
            a = Ai[p]
            if a != 0.0:
                Bp = B[p]
                for j in range(n):
                    oi[j] += a * Bp[j]
    return out


def _transpose(A: List[List[float]]) -> List[List[float]]:
    return [list(row) for row in zip(*A)]


def newton_schulz_orthogonalize(G: List[List[float]], steps: int = 5) -> List[List[float]]:
    """Nearest-orthogonal approximation of G via Newton-Schulz (Muon core).

    Coefficients (3.4445, -4.7750, 2.0315) from Keller Jordan's Muon work;
    5 steps suffice at these model scales. G normalized by Frobenius norm
    first so the iteration converges. Pure Python: only viable for small
    matrices (see MuonOptimizer's `max_dim` cap).
    """
    m, n = len(G), len(G[0])
    norm = math.sqrt(sum(v * v for row in G for v in row))
    if norm == 0.0:
        return [row[:] for row in G]
    X = [[v / (norm + 1e-7) for v in row] for row in G]
    a, b, c = 3.4445, -4.7750, 2.0315
    for _ in range(steps):
        Xt = _transpose(X)
        A = _matmul(X, Xt)          # m×m
        A2 = _matmul(A, A)          # m×m
        # B = b*A + c*A2 ; X = a*X + B@X
        B = [[b * x + c * y for x, y in zip(ra, rb)] for ra, rb in zip(A, A2)]
        BX = _matmul(B, X)          # m×n
        X = [[a * x + y for x, y in zip(rx, rb)] for rx, rb in zip(X, BX)]
    return X


class MuonOptimizer:
    """Muon + AdamW hybrid (2025 SOTA for LLM loss reduction, stdlib port).

    2D weight matrices at or below `max_dim` get Muon updates (momentum +
    Newton-Schulz orthogonalization, scaled by `sqrt(max_dim)` like the
    reference implementation); everything else (embeddings, heads, biases,
    oversized matrices) falls back to an internal AdamW — the same
    "Muon for hidden layers, AdamW for the rest" hybrid the literature
    converged on. Same `step()` contract as Adam/Lion.
    """

    def __init__(self, params: List[Value], tensors=None, lr: float = 0.01,
                 beta: float = 0.95, adam_betas: Tuple[float, float] = (0.9, 0.999),
                 eps: float = 1e-8, weight_decay: float = 0.0,
                 lr_scale: float = 1.0, muon_scale: float = 1.0,
                 max_dim: int = 64, ns_steps: int = 5):
        self.params = params
        self.lr = lr
        self.beta = beta
        self.eps = eps
        self.weight_decay = weight_decay
        self.lr_scale = lr_scale
        self.muon_scale = muon_scale
        self.max_dim = max_dim
        self.ns_steps = ns_steps
        # Map flat indices -> muon matrices (list of (rows, cols, [flat idx])).
        self.groups: List[Tuple[int, int, List[int]]] = []
        if tensors:
            pos = {id(p): k for k, p in enumerate(params)}
            for rows in tensors.values():
                try:
                    r, c = len(rows), len(rows[0]) if rows else 0
                except Exception:
                    continue
                if r > 1 and c > 1 and max(r, c) <= max_dim:
                    idx = [pos.get(id(v)) for row in rows for v in row]
                    if len(idx) == r * c and all(k is not None for k in idx):
                        self.groups.append((r, c, idx))
        self.muon_mom = [[[0.0] * c for _ in range(r)] for r, c, _ in self.groups]
        self.muon_idx = set(k for _, _, idx in self.groups for k in idx)
        self.adam = AdamOptimizer(params, lr=lr, betas=adam_betas, eps=eps,
                                  weight_decay=weight_decay, lr_scale=lr_scale)

    def step(self, indices=None):
        """One hybrid step (Adam subset support included)."""
        allow = set(indices) if indices is not None else None
        rest = [i for i in range(len(self.params))
                if i not in self.muon_idx and (allow is None or i in allow)]
        if rest:
            self.adam.lr = self.lr
            self.adam.step(indices=rest)
        base = self.lr * self.adam.lr_scale
        for (r, c, idx), mom in zip(
                [(rr, cc, ii) for rr, cc, ii in self.groups
                 if allow is None or all(k in allow for k in ii)],
                [m for (rr, cc, ii), m in zip(self.groups, self.muon_mom)
                 if allow is None or all(k in allow for k in ii)]):
            # Stage grads into momentum buffer.
            k = 0
            for i in range(r):
                for j in range(c):
                    g = self.params[idx[k]].grad
                    mom[i][j] = self.beta * mom[i][j] + (1.0 - self.beta) * g
                    k += 1
            ortho = newton_schulz_orthogonalize(mom, self.ns_steps)
            step_lr = base * math.sqrt(max(r, c)) * self.muon_scale
            k = 0
            for i in range(r):
                for j in range(c):
                    p = self.params[idx[k]]
                    p.data -= step_lr * (ortho[i][j] + self.weight_decay * p.data)
                    p.grad = 0.0
                    k += 1


# ---- Checkpointing ----------------------------------------------------------
