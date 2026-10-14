"""RWKV GPU acceleration (GTX 970+ compatible, compute 5.2).

Canonical GPU path for the RWKV architecture (``tinylm/models_linear.py`` +
``csrc/rwkv_kernel.cu``). The legacy MLP path lives in ``accel_cuda.py``.

Entry points raise ``RuntimeError`` when the library is missing or the
kernel reports an error, so callers always fall back to CPU.
"""
import ctypes
import math
import os
from typing import List, Optional


def _lib_path() -> str:
    """Find the RWKV CUDA shared library."""
    base = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(base, "csrc", "libtinylm_rwkv.so"),
        os.path.join(base, "libtinylm_rwkv.so"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError("libtinylm_rwkv.so not found. Run ./build.sh")


def _load_lib():
    """Load the RWKV CUDA library (cached)."""
    global _rwkv_lib
    if _rwkv_lib is not None:
        return _rwkv_lib
    lib = ctypes.CDLL(_lib_path())
    lib.rwkv_init_buffers.argtypes = [
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ]
    lib.rwkv_init_buffers.restype = ctypes.c_int
    lib.rwkv_reset_state.argtypes = []
    lib.rwkv_reset_state.restype = None
    lib.rwkv_train_step.argtypes = [
        ctypes.POINTER(ctypes.c_double),  # params (in/out)
        ctypes.POINTER(ctypes.c_int),     # windows
        ctypes.POINTER(ctypes.c_int),     # negatives (or NULL)
        ctypes.c_int,                     # batch
        ctypes.c_int,                     # ctx
        ctypes.c_int,                     # vocab
        ctypes.c_int,                     # emb
        ctypes.c_int,                     # hidden
        ctypes.c_int,                     # n_layers
        ctypes.c_int,                     # has_in_proj
        ctypes.c_int,                     # n_negatives
        ctypes.c_double,                  # lr
        ctypes.c_double,                  # momentum
        ctypes.c_ulonglong,               # seed
        ctypes.POINTER(ctypes.c_double),  # out_loss (averaged)
        ctypes.c_double,                  # max_norm (clip, 0 = off)
    ]
    lib.rwkv_train_step.restype = ctypes.c_int
    lib.rwkv_batch_loss_grads.argtypes = [
        ctypes.POINTER(ctypes.c_double),  # params (read-only)
        ctypes.POINTER(ctypes.c_int),     # windows
        ctypes.POINTER(ctypes.c_int),     # negatives (or NULL)
        ctypes.c_int,                     # batch
        ctypes.c_int,                     # ctx
        ctypes.c_int,                     # vocab
        ctypes.c_int,                     # emb
        ctypes.c_int,                     # hidden
        ctypes.c_int,                     # n_layers
        ctypes.c_int,                     # has_in_proj
        ctypes.c_int,                     # n_negatives
        ctypes.c_ulonglong,               # seed
        ctypes.POINTER(ctypes.c_double),  # out_loss (summed)
        ctypes.POINTER(ctypes.c_double),  # out_grads (summed)
        ctypes.c_double,                  # max_norm (informational)
    ]
    lib.rwkv_batch_loss_grads.restype = ctypes.c_int
    lib.rwkv_generate_step.argtypes = [
        ctypes.POINTER(ctypes.c_double),  # params (read-only)
        ctypes.POINTER(ctypes.c_int),     # input_ids
        ctypes.c_int,                     # seq_len
        ctypes.c_int,                     # vocab
        ctypes.c_int,                     # emb
        ctypes.c_int,                     # hidden
        ctypes.c_int,                     # n_layers
        ctypes.c_int,                     # has_in_proj
        ctypes.c_int,                     # ctx (unused)
        ctypes.c_ulonglong,               # seed (unused)
        ctypes.POINTER(ctypes.c_double),  # out_logits
        ctypes.POINTER(ctypes.c_double),  # states (unused, NULL ok)
        ctypes.c_int,                     # max_seq_len (unused)
    ]
    lib.rwkv_generate_step.restype = ctypes.c_int
    _rwkv_lib = lib
    return _rwkv_lib


_rwkv_lib = None


def _rwkv_available() -> bool:
    """True when the RWKV CUDA library loads."""
    try:
        return _load_lib() is not None
    except Exception:
        return False


def device_info() -> dict:
    """Report RWKV GPU backend status for UI/diagnostics."""
    try:
        path = _lib_path()
    except FileNotFoundError:
        return {"backend": "rwkv-cuda", "available": False, "reason": "libtinylm_rwkv.so not built"}
    try:
        _load_lib()
    except Exception as exc:
        return {"backend": "rwkv-cuda", "available": False, "reason": str(exc), "lib": path}
    return {"backend": "rwkv-cuda", "available": True, "lib": path,
            "note": "full WKV forward+backward training kernel (SGD+momentum)"}


def _dims_of(vocab: int, emb: int, hidden: int, n_layers: int):
    has_in = 1 if emb != hidden else 0
    return has_in


def rwkv_reset_state() -> None:
    """Reset momentum velocity (no-op when unavailable)."""
    if _rwkv_available():
        _load_lib().rwkv_reset_state()


def _check_layout(flat: List[float], vocab: int, emb: int, hidden: int,
                  n_layers: int) -> int:
    """Validate the flat vector against the RWKVModel.params layout."""
    has_in = _dims_of(vocab, emb, hidden, n_layers)
    n = vocab * emb
    if has_in:
        n += hidden * emb + hidden
    n += n_layers * (9 * hidden * hidden + 13 * hidden)
    n += vocab * hidden + vocab
    if len(flat) != n:
        raise RuntimeError(
            f"RWKV flat layout mismatch: got {len(flat)}, expected {n} "
            f"(vocab={vocab} emb={emb} hidden={hidden} layers={n_layers})")
    return has_in


def rwkv_train_step(flat: List[float], windows: List[int],
                    vocab: int, emb: int, hidden: int, n_layers: int, ctx: int,
                    n_negatives: int, lr: float, seed: int,
                    momentum: float = 0.0, max_norm: float = 0.0,
                    negatives: Optional[List[int]] = None) -> tuple:
    """Single RWKV training step on GPU (averaged loss + SGD/momentum update).

    Mirrors ``accel_cuda.cuda_train_step`` semantics: ``windows`` holds
    ``batch`` groups of ``ctx`` context ids + 1 target id; grads are
    averaged over the batch, clipped by global norm when ``max_norm`` > 0,
    then applied with ``v = momentum*v + g; p -= lr*v`` (persistent
    velocity, reset with :func:`rwkv_reset_state`).

    Returns ``(loss, updated_flat_params)``. Raises ``RuntimeError`` when
    unavailable or the kernel errors so callers fall back to CPU.
    """
    if not _rwkv_available():
        raise RuntimeError("RWKV CUDA acceleration not available")
    has_in = _check_layout(flat, vocab, emb, hidden, n_layers)
    batch = len(windows) // (ctx + 1)
    if batch < 1:
        raise RuntimeError("empty batch")
    lib = _load_lib()
    lib.rwkv_init_buffers(vocab, emb, hidden, n_layers, has_in, ctx, batch)
    params_arr = (ctypes.c_double * len(flat))(*flat)
    windows_arr = (ctypes.c_int * len(windows))(*windows)
    negs_arr = None
    if negatives:
        negs_arr = (ctypes.c_int * len(negatives))(*negatives)
    out_loss = ctypes.c_double(0.0)
    rc = lib.rwkv_train_step(
        params_arr, windows_arr, negs_arr, batch, ctx, vocab, emb, hidden,
        n_layers, has_in, n_negatives, lr, momentum, seed,
        ctypes.byref(out_loss), max_norm,
    )
    if rc != 0:
        raise RuntimeError(f"RWKV CUDA train_step failed (rc={rc})")
    return out_loss.value, [params_arr[i] for i in range(len(flat))]


def rwkv_batch_grads(flat: List[float], windows: List[int],
                     vocab: int, emb: int, hidden: int, n_layers: int, ctx: int,
                     n_negatives: int, seed: int = 0,
                     negatives: Optional[List[int]] = None,
                     max_norm: float = 0.0) -> tuple:
    """Summed loss + summed grads for one batch on the GPU (no update).

    Mirrors ``accel_cuda.cuda_batch_grads`` semantics: grads and loss are
    SUMMED (caller divides). Gradient clipping (max_norm) is applied on the
    host so semantics match the CPU path exactly. Raises ``RuntimeError``
    when unavailable or the kernel errors, so callers fall back to CPU.
    """
    if not _rwkv_available():
        raise RuntimeError("RWKV CUDA acceleration not available")
    has_in = _check_layout(flat, vocab, emb, hidden, n_layers)
    batch = len(windows) // (ctx + 1)
    if batch < 1:
        raise RuntimeError("empty batch")
    lib = _load_lib()
    params_arr = (ctypes.c_double * len(flat))(*flat)
    windows_arr = (ctypes.c_int * len(windows))(*windows)
    negs_arr = None
    if negatives:
        negs_arr = (ctypes.c_int * len(negatives))(*negatives)
    out_loss = ctypes.c_double(0.0)
    out_grads = (ctypes.c_double * len(flat))()
    rc = lib.rwkv_batch_loss_grads(
        params_arr, windows_arr, negs_arr, batch, ctx, vocab, emb, hidden,
        n_layers, has_in, n_negatives, seed,
        ctypes.byref(out_loss), out_grads, 0.0,
    )
    if rc != 0:
        raise RuntimeError(f"RWKV CUDA batch grads failed (rc={rc})")
    grads = [out_grads[i] for i in range(len(flat))]
    if max_norm > 0.0:
        norm = math.sqrt(sum(g * g for g in grads))
        if norm > max_norm and norm > 0.0:
            scale = max_norm / norm
            grads = [g * scale for g in grads]
    return out_loss.value, grads


def rwkv_generate_step(flat: List[float], input_ids: List[int],
                       vocab: int, emb: int, hidden: int, n_layers: int, ctx: int,
                       seed: int = 0) -> List[float]:
    """Single RWKV generation step (logits out).

    Exact float-order forward reference (runs on the host inside the compiled
    library: single-sequence generation is transfer-bound, so device offload
    would be slower). Raises ``RuntimeError`` when unavailable so the caller
    falls back to ``RWKVModel.fast_logits`` on CPU.
    """
    if not _rwkv_available():
        raise RuntimeError("RWKV CUDA acceleration not available")
    has_in = _check_layout(flat, vocab, emb, hidden, n_layers)
    if not input_ids:
        raise RuntimeError("empty input")
    lib = _load_lib()
    params_arr = (ctypes.c_double * len(flat))(*flat)
    ids_arr = (ctypes.c_int * len(input_ids))(*input_ids)
    out_logits = (ctypes.c_double * vocab)()
    rc = lib.rwkv_generate_step(
        params_arr, ids_arr, len(input_ids), vocab, emb, hidden,
        n_layers, has_in, ctx, seed, out_logits, None, len(input_ids),
    )
    if rc != 0:
        raise RuntimeError(f"RWKV CUDA generate_step failed (rc={rc})")
    return [out_logits[i] for i in range(vocab)]
