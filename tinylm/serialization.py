"""4-bit model serialization, vocab growth, param counting (stdlib only)."""
import random
import struct
from pathlib import Path

from tinylm.autograd import Value

MAGIC = b"TLM4"   # file signature: TinyLM, 4-bit


def sync_vocab(model, vocab_size, seed=None):
    """Grow the model's embedding, output and bias rows to vocab_size.
    Existing rows are kept, and the model is never recreated or shrunk."""
    old = len(model.C)
    if vocab_size <= old:
        return False
    rng = random.Random(seed)
    emb, hidden = len(model.C[0]), len(model.b1)
    extra = vocab_size - old
    model.C += [[Value(rng.gauss(0.0, 0.1)) for _ in range(emb)] for _ in range(extra)]
    model.W2 += [[Value(rng.gauss(0.0, 0.1)) for _ in range(hidden)] for _ in range(extra)]
    model.b2 += [Value(0.0) for _ in range(extra)]
    model.params = (
        [p for row in model.C for p in row]
        + [p for row in model.W1 for p in row] + model.b1
        + [p for row in model.W2 for p in row] + model.b2
    )
    return True

def load_corpus(folder, pattern="*.txt"):
    """Read every matching file under `folder` and join them into one string."""
    files = sorted(Path(folder).rglob(pattern))
    if not files:
        raise FileNotFoundError(f"No {pattern} files found in {folder}")
    parts = []
    for path in files:
        text = path.read_text(encoding="utf-8", errors="ignore")
        parts.append(text)
        print(f"read {path} ({len(text)} characters)")
    return "\n".join(parts)


def _quantize_rows(rows):
    """Quantize a 2D list of floats to 4-bit signed integers, one scale per row.

    Returns (scales, packed_bytes). Two 4-bit values share each byte.
    """
    scales, flat = [], []
    for row in rows:
        values = [v.data for v in row]
        max_abs = max(abs(x) for x in values) or 1e-8
        scale = max_abs / 7.0                      # map to [-7, 7]
        scales.append(scale)
        for x in values:
            q = max(-7, min(7, round(x / scale)))
            flat.append(q + 8)                     # shift to [1, 15], fits 4 bits

    if len(flat) % 2:
        flat.append(8)                             # pad with zero
    packed = bytes((flat[i] << 4) | flat[i + 1] for i in range(0, len(flat), 2))
    return scales, packed


def _dequantize_rows(scales, packed, n_rows, n_cols):
    """Inverse of _quantize_rows: returns a 2D list of floats."""
    nibbles = []
    for byte in packed:
        nibbles.append(byte >> 4)
        nibbles.append(byte & 0x0F)

    out, idx = [], 0
    for r in range(n_rows):
        scale = scales[r]
        row = []
        for _ in range(n_cols):
            row.append((nibbles[idx] - 8) * scale)
            idx += 1
        out.append(row)
    return out


def save_model(model, path):
    """Save architecture and 4-bit quantized weights to a binary file.

    Works for any model that implements:
      - model.get_tensors() -> Dict[str, List[List[Value]]]
      - model.get_config() -> Dict[str, Any] (must include 'arch' for identification)

    Layout (little-endian):
      magic "TLM4"
      uint32: config_json_length
      config_json (UTF-8)
      for each tensor in get_tensors().values():
        uint32 rows, uint32 cols
        float32 scale per row
        packed 4-bit values, (rows * cols + 1) // 2 bytes
    """
    import json
    config = model.get_config()
    config_json = json.dumps(config, separators=(',', ':')).encode('utf-8')

    with open(path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<I", len(config_json)))
        f.write(config_json)

        tensors = model.get_tensors()
        for name, rows in tensors.items():
            n_rows, n_cols = len(rows), len(rows[0])
            scales, packed = _quantize_rows(rows)
            f.write(struct.pack("<2I", n_rows, n_cols))
            f.write(struct.pack(f"<{n_rows}f", *scales))
            f.write(packed)


def load_model(path, model_factory=None):
    """Rebuild a model from a file written by save_model.

    Args:
        path: Path to model file
        model_factory: Optional callable(config) -> model instance.
                       If None, tries to use registered model factory for config['arch'].

    Returns:
        Model instance with loaded weights.
    """
    import json
    with open(path, "rb") as f:
        if f.read(4) != MAGIC:
            raise ValueError(f"{path} is not a 4-bit TinyLM file")
        config_len = struct.unpack("<I", f.read(4))[0]
        config_json = f.read(config_len).decode('utf-8')
        config = json.loads(config_json)

        # Create model instance
        if model_factory is not None:
            model = model_factory(config)
        else:
            # Try to get from registry
            arch = config.get('arch', 'TinyLM')
            try:
                from registry import get_model_factory
                factory = get_model_factory(arch)
                # Filter out 'arch' from config for factory
                model_config = {k: v for k, v in config.items() if k != 'arch'}
                model = factory(**model_config)
            except (ImportError, KeyError):
                # Fallback to TinyLM
                from tinylm.models_tiny import TinyLM
                model_config = {k: v for k, v in config.items() if k != 'arch'}
                model = TinyLM(**model_config)

        # Load tensors
        tensors = model.get_tensors()
        for name, target in tensors.items():
            n_rows, n_cols = struct.unpack("<2I", f.read(8))
            if n_rows != len(target) or n_cols != len(target[0]):
                raise ValueError(f"Shape mismatch for tensor '{name}': "
                                 f"expected ({len(target)}, {len(target[0])}), "
                                 f"got ({n_rows}, {n_cols})")
            scales = struct.unpack(f"<{n_rows}f", f.read(4 * n_rows))
            packed = f.read((n_rows * n_cols + 1) // 2)
            values = _dequantize_rows(scales, packed, n_rows, n_cols)
            for row, new_row in zip(target, values):
                for v, x in zip(row, new_row):
                    v.data = x

    return model


def count_params(model) -> int:
    """Count total parameters in a model.

    Works for any model that implements model.get_tensors() -> Dict[str, List[List[Value]]].
    """
    total = 0
    for rows in model.get_tensors().values():
        for row in rows:
            total += len(row)
    return total


def enforce_param_limit(model, max_params: int, model_name: str = "model"):
    """Enforce parameter limit at init. Raises ValueError if exceeded."""
    n = count_params(model)
    if n > max_params:
        raise ValueError(f"{model_name} has {n:,} parameters, exceeds limit of {max_params:,}")
    return n
