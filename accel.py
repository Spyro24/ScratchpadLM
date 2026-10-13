"""Optional C acceleration for training, loaded via ctypes (stdlib only).

Compiles from csrc/train_kernel.c to libtinylm_accel.so (see build.sh).
If the shared library is missing or fails to load, `AVAILABLE` is False and
ai.train()/train_parallel() fall back to the pure-Python autograd path.

The C kernels are numerically equivalent to TinyLM.sampled_loss + backward()
+ in-place SGD (double precision throughout).
"""
import ctypes
import os
from pathlib import Path
from typing import List, Optional, Tuple

LIB_PATH = Path(__file__).parent / "libtinylm_accel.so"

_lib = None
AVAILABLE = False

if LIB_PATH.exists():
    try:
        _lib = ctypes.CDLL(str(LIB_PATH))

        _dbl_p = ctypes.POINTER(ctypes.c_double)
        _int_p = ctypes.POINTER(ctypes.c_int)

        _lib.tl_batch_loss_grads.argtypes = [
            _dbl_p,          # params
            _int_p,          # windows
            _int_p,          # negatives (or NULL)
            ctypes.c_int,    # batch
            ctypes.c_int,    # ctx
            ctypes.c_int,    # vocab
            ctypes.c_int,    # emb
            ctypes.c_int,    # hidden
            ctypes.c_int,    # n_negatives
            ctypes.c_ulonglong,  # seed
            _dbl_p,          # out_loss
            _dbl_p,          # out_grads
            ctypes.c_double, # max_norm
        ]
        _lib.tl_batch_loss_grads.restype = ctypes.c_int

        _lib.tl_train_step.argtypes = [
            _dbl_p,          # params (in-place)
            _int_p,          # windows
            ctypes.c_int,    # batch
            ctypes.c_int,    # ctx
            ctypes.c_int,    # vocab
            ctypes.c_int,    # emb
            ctypes.c_int,    # hidden
            ctypes.c_int,    # n_negatives
            ctypes.c_double, # lr
            ctypes.c_double, # momentum
            ctypes.c_ulonglong,  # seed
            _dbl_p,          # out_loss
            ctypes.c_double, # max_norm
        ]
        _lib.tl_train_step.restype = ctypes.c_int

        _lib.tl_reset_velocity.argtypes = []
        _lib.tl_reset_velocity.restype = None

        AVAILABLE = True
    except OSError:
        _lib = None
        AVAILABLE = False


class CAccelError(RuntimeError):
    """Raised when a C kernel call fails."""


_ERR = {
    -1: "invalid argument",
    -2: "token id out of range",
    -3: "out of memory",
}


def _as_dbl_array(values) -> ctypes.Array:
    return (ctypes.c_double * len(values))(*values)


def _as_int_array(values) -> ctypes.Array:
    return (ctypes.c_int * len(values))(*values)


def batch_loss_grads(flat: List[float], windows: List[int],
                      dims: Tuple[int, int, int, int], n_negatives: int,
                      seed: int = 0,
                      negatives: Optional[List[int]] = None,
                      max_norm: float = 0.0) -> Tuple[float, List[float]]:
    """Summed loss and summed grads for a batch (no /batch scaling — matches
    summing per-window autograd losses + one backward()). `flat` is TinyLM
    params in order, `windows` is [ctx ids..., target] per window flattened.
    `negatives` (flat, batch*n_negatives) or internal RNG with `seed` selects
    the sampled-softmax candidates.
    If max_norm > 0, gradients are clipped by global norm before returning."""
    vocab, ctx, emb, hidden = dims
    batch = len(windows) // (ctx + 1)
    if batch < 1:
        raise CAccelError("empty batch")

    arr_flat = _as_dbl_array(flat)
    arr_win = _as_int_array(windows)
    arr_neg = _as_int_array(negatives) if negatives else ctypes.POINTER(ctypes.c_int)()
    arr_loss = ctypes.c_double()
    arr_grads = (ctypes.c_double * len(flat))()

    rc = _lib.tl_batch_loss_grads(
        arr_flat, arr_win, arr_neg,
        batch, ctx, vocab, emb, hidden,
        n_negatives, seed,
        ctypes.byref(arr_loss), arr_grads,
        ctypes.c_double(max_norm),
    )
    if rc != 0:
        raise CAccelError(f"tl_batch_loss_grads failed: {_ERR.get(rc, rc)}")
    return arr_loss.value, list(arr_grads)


def train_step(flat: List[float], windows: List[int],
               dims: Tuple[int, int, int, int], n_negatives: int,
               lr: float, seed: int = 0,
               momentum: float = 0.0,
               max_norm: float = 0.0) -> Tuple[float, List[float]]:
    """One fused step: forward + backward + SGD on a copy of `flat`.
    Returns (mean_loss, updated_flat).
    If max_norm > 0, the batch-averaged gradient is clipped by global norm
    before the update. `momentum` enables SGD with momentum (velocity is kept
    inside the kernel across steps; call reset_momentum() to start a new run).
    """
    vocab, ctx, emb, hidden = dims
    batch = len(windows) // (ctx + 1)
    if batch < 1:
        raise CAccelError("empty batch")

    arr_flat = _as_dbl_array(flat)
    arr_win = _as_int_array(windows)
    arr_loss = ctypes.c_double()

    rc = _lib.tl_train_step(
        arr_flat, arr_win,
        batch, ctx, vocab, emb, hidden,
        n_negatives, lr, momentum, seed,
        ctypes.byref(arr_loss),
        ctypes.c_double(max_norm),
    )
    if rc != 0:
        raise CAccelError(f"tl_train_step failed: {_ERR.get(rc, rc)}")
    return arr_loss.value, list(arr_flat)


def reset_momentum() -> None:
    """Zero the C kernel's momentum state for the calling thread.

    Call once at the start of a training run (see `_train_c`) so a new model
    never starts with velocity left over from an earlier run.
    """
    if _lib is None:
        return
    _lib.tl_reset_velocity()