"""CUDA acceleration for training (optional, GTX 970 compatible).

LEGACY path for the old MLP arch (``tinylm/models_tiny.py``). Canonical going
forward is the RWKV arch (``tinylm/models_linear.py`` + ``accel_rwkv.py``).
Kept as fallback until the old arch is removed.
"""
import ctypes
import os
from typing import List, Optional


def _lib_path() -> str:
    """Find the CUDA shared library."""
    base = os.path.dirname(__file__)
    candidates = [
        os.path.join(base, "libtinylm_cuda.so"),
        os.path.join(base, "csrc", "libtinylm_cuda.so"),
        os.path.join(base, "..", "csrc", "libtinylm_cuda.so"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError("libtinylm_cuda.so not found. Run ./build_cuda.sh")


def _load_lib():
    """Load the CUDA library."""
    global _cuda_lib
    if '_cuda_lib' in globals() and _cuda_lib is not None:
        return _cuda_lib
    try:
        _cuda_lib = ctypes.CDLL(_lib_path())
        # Define function signatures
        _cuda_lib.cuda_init_buffers.argtypes = [
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_int
        ]
        _cuda_lib.cuda_init_buffers.restype = ctypes.c_int
        
        _cuda_lib.cuda_reset_velocity.argtypes = []
        _cuda_lib.cuda_reset_velocity.restype = None
        
        _cuda_lib.cuda_train_step.argtypes = [
            ctypes.POINTER(ctypes.c_double),  # params
            ctypes.POINTER(ctypes.c_int),     # windows
            ctypes.POINTER(ctypes.c_int),     # negatives
            ctypes.c_int,                     # batch
            ctypes.c_int,                     # ctx
            ctypes.c_int,                     # vocab
            ctypes.c_int,                     # emb
            ctypes.c_int,                     # hidden
            ctypes.c_int,                     # n_negatives
            ctypes.c_double,                  # lr
            ctypes.c_double,                  # momentum
            ctypes.c_ulonglong,               # seed
            ctypes.POINTER(ctypes.c_double),  # out_loss
            ctypes.c_double,                  # max_norm
        ]
        _cuda_lib.cuda_train_step.restype = ctypes.c_int

        _cuda_lib.cuda_batch_loss_grads.argtypes = [
            ctypes.POINTER(ctypes.c_double),  # params (read-only)
            ctypes.POINTER(ctypes.c_int),     # windows
            ctypes.POINTER(ctypes.c_int),     # negatives (or NULL)
            ctypes.c_int,                     # batch
            ctypes.c_int,                     # ctx
            ctypes.c_int,                     # vocab
            ctypes.c_int,                     # emb
            ctypes.c_int,                     # hidden
            ctypes.c_int,                     # n_negatives
            ctypes.c_ulonglong,               # seed
            ctypes.POINTER(ctypes.c_double),  # out_loss (summed)
            ctypes.POINTER(ctypes.c_double),  # out_grads (summed)
            ctypes.c_double,                  # max_norm (informational; clipped on host)
        ]
        _cuda_lib.cuda_batch_loss_grads.restype = ctypes.c_int
        return _cuda_lib
    except Exception as e:
        print(f"CUDA acceleration unavailable: {e}")
        return None


_cuda_lib = None


def _cuda_available() -> bool:
    """Check if CUDA acceleration is available."""
    try:
        return _load_lib() is not None
    except Exception:
        return False


def device_info() -> dict:
    """Report legacy CUDA backend status for UI/diagnostics."""
    try:
        path = _lib_path()
    except FileNotFoundError:
        return {"backend": "legacy-cuda", "available": False, "reason": "libtinylm_cuda.so not built"}
    try:
        _load_lib()
    except Exception as exc:
        return {"backend": "legacy-cuda", "available": False, "reason": str(exc), "lib": path}
    return {"backend": "legacy-cuda", "available": True, "lib": path,
            "note": "MLP arch only; deprecated in favor of accel_rwkv"}


def cuda_train_step(flat: List[float], windows: List[int], dims: tuple,
                    n_negatives: int, lr: float, seed: int,
                    momentum: float = 0.0, max_norm: float = 0.0,
                    negatives: Optional[List[int]] = None,
                    max_n_negatives: Optional[int] = None) -> tuple:
    """CUDA-accelerated training step.
    
    Args:
        flat: Flattened model parameters (double)
        windows: Training windows (int)
        dims: (vocab, ctx, emb, hidden)
        n_negatives: Number of negative samples (current step)
        lr: Learning rate
        seed: Random seed
        momentum: Momentum factor (0 = plain SGD)
        max_norm: Gradient clipping threshold (0 = disabled)
        negatives: Pre-sampled negative token IDs (optional)
        max_n_negatives: Maximum n_negatives for buffer allocation (handles adaptive n_negatives)
    
    Returns:
        (loss, updated_flat_params)
    """
    if not _cuda_available():
        raise RuntimeError("CUDA acceleration not available")
    
    vocab, ctx, emb, hidden = dims
    batch = len(windows) // (ctx + 1)

    lib = _load_lib()

    # Initialize persistent buffers (velocity + scratch sized for the
    # adaptive maximum when given).
    init_n_negatives = max_n_negatives if max_n_negatives is not None else n_negatives
    lib.cuda_init_buffers(ctx, vocab, emb, hidden, init_n_negatives, batch)

    # Prepare arrays
    params_arr = (ctypes.c_double * len(flat))(*flat)
    windows_arr = (ctypes.c_int * len(windows))(*windows)
    
    negs_arr = None
    if negatives:
        negs_arr = (ctypes.c_int * len(negatives))(*negatives)
    
    out_loss = ctypes.c_double(0.0)
    
    # Call CUDA kernel
    rc = lib.cuda_train_step(
        params_arr, windows_arr, negs_arr,
        batch, ctx, vocab, emb, hidden,
        n_negatives, lr, momentum, seed,
        ctypes.byref(out_loss), max_norm
    )
    
    if rc != 0:
        raise RuntimeError(f"CUDA training step failed with code {rc}")
    
    # Return loss and updated params
    updated = [params_arr[i] for i in range(len(flat))]
    return out_loss.value, updated


def cuda_reset_velocity():
    """Reset momentum velocity."""
    if _cuda_available():
        lib = _load_lib()
        lib.cuda_reset_velocity()


def cuda_batch_grads(flat: List[float], windows: List[int], dims: tuple,
                     n_negatives: int, seed: int = 0,
                     negatives: Optional[List[int]] = None,
                     max_norm: float = 0.0) -> tuple:
    """Summed loss + summed grads for one batch on the GPU (no update).

    Mirrors accel.batch_loss_grads semantics: grads and loss are SUMMED
    (caller divides). The kernel parallelizes across windows (grid-stride
    loop, global atomics) with per-thread scratch sized from the call's own
    dims. Raises RuntimeError when the library is missing or the device
    errors, so callers fall back to CPU. Gradient clipping (max_norm) is
    applied on the host so semantics match the C path exactly.
    """
    import math
    if not _cuda_available():
        raise RuntimeError("CUDA acceleration not available")
    vocab, ctx, emb, hidden = dims
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

    rc = lib.cuda_batch_loss_grads(
        params_arr, windows_arr, negs_arr,
        batch, ctx, vocab, emb, hidden,
        n_negatives, seed,
        ctypes.byref(out_loss), out_grads, 0.0,
    )
    if rc != 0:
        raise RuntimeError(f"CUDA batch grads failed with code {rc}")

    grads = [out_grads[i] for i in range(len(flat))]
    if max_norm > 0.0:
        norm = math.sqrt(sum(g * g for g in grads))
        if norm > max_norm and norm > 0.0:
            scale = max_norm / norm
            grads = [g * scale for g in grads]
    return out_loss.value, grads


def cuda_train_steps(flat: List[float], all_ids: List[int], steps: int,
                     batch_size: int, n_negatives: int, lr: float, seed: int,
                     momentum: float = 0.0, ctx: int = 0, vocab: int = 0,
                     emb: int = 0, hidden: int = 0, checkpoint_every: int = 0,
                     eval_every: int = 0, val_split: float = 0.0,
                     val_ids: Optional[List[int]] = None) -> tuple:
    """CUDA-accelerated multi-step training.
    
    Runs multiple training steps on GPU, only syncing for checkpoints/validation.
    
    Args:
        flat: Flattened model parameters (double)
        all_ids: All training token IDs (int)
        steps: Number of training steps
        batch_size: Batch size
        n_negatives: Number of negative samples
        lr: Learning rate
        seed: Random seed
        momentum: Momentum factor (0 = plain SGD)
        ctx: Context length
        vocab: Vocabulary size
        emb: Embedding dimension
        hidden: Hidden dimension
        checkpoint_every: Checkpoint frequency (0 = off)
        eval_every: Evaluation frequency (0 = off)
        val_split: Validation split ratio
        val_ids: Optional validation token IDs
        
    Returns:
        (final_loss, steps_done, updated_flat_params)
    """
    if not _cuda_available():
        raise RuntimeError("CUDA acceleration not available")
    
    ids_len = len(all_ids)
    val_ids_len = len(val_ids) if val_ids else 0
    
    lib = _load_lib()
    
    # Initialize buffers
    lib.cuda_init_buffers(ctx, vocab, emb, hidden, n_negatives, batch_size)
    
    # Prepare arrays
    params_arr = (ctypes.c_double * len(flat))(*flat)
    all_ids_arr = (ctypes.c_int * len(all_ids))(*all_ids)
    
    val_ids_arr = None
    if val_ids:
        val_ids_arr = (ctypes.c_int * len(val_ids))(*val_ids)
    
    out_final_loss = ctypes.c_double(0.0)
    out_steps_done = ctypes.c_int(0)
    
    # Call CUDA multi-step training
    rc = lib.cuda_train_steps(
        params_arr, all_ids_arr, ids_len,
        steps, batch_size, n_negatives,
        lr, momentum, seed,
        ctx, vocab, emb, hidden,
        checkpoint_every, eval_every,
        ctypes.byref(out_final_loss), ctypes.byref(out_steps_done),
        val_split, val_ids_arr, val_ids_len
    )
    
    if rc != 0:
        raise RuntimeError(f"CUDA multi-step training failed with code {rc}")
    
    updated = [params_arr[i] for i in range(len(flat))]
    return out_final_loss.value, out_steps_done.value, updated