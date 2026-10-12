import math
import random
from dataclasses import dataclass
from typing import List, Optional

from tokenizer import BPETokenizer

import struct
from multiprocessing import Pool, cpu_count

try:
    import accel
except Exception:  # pragma: no cover - accel never raises, belt and braces
    accel = None

MAGIC = b"TLM4"   # file signature: TinyLM, 4-bit

from pathlib import Path


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
                from ai import TinyLM
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

import operator


# ---- autograd -------------------------------------------------------------

class Value:
    """A scalar that remembers how it was computed, so gradients can flow back."""
    __slots__ = ("data", "grad", "_prev", "_back")

    def __init__(self, data, prev=(), back=None):
        self.data = float(data)
        self.grad = 0.0
        self._prev = prev
        self._back = back          # function that pushes out.grad into the inputs

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other))

        def back():
            self.grad += out.grad
            other.grad += out.grad
        out._back = back
        return out

    __radd__ = __add__

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other))

        def back():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._back = back
        return out

    __rmul__ = __mul__

    def __neg__(self):
        return self * -1.0

    def __sub__(self, other):
        return self + (-other)

    def __pow__(self, n):
        out = Value(self.data ** n, (self,))

        def back():
            self.grad += n * self.data ** (n - 1) * out.grad
        out._back = back
        return out

    def __truediv__(self, other):
        return self * other ** -1

    def exp(self):
        e = math.exp(self.data)
        out = Value(e, (self,))

        def back():
            self.grad += e * out.grad
        out._back = back
        return out

    def log(self):
        out = Value(math.log(self.data), (self,))

        def back():
            self.grad += out.grad / self.data
        out._back = back
        return out

    def tanh(self):
        t = math.tanh(self.data)
        out = Value(t, (self,))

        def back():
            self.grad += (1.0 - t * t) * out.grad
        out._back = back
        return out

    def backward(self):
        # Iterative post-order traversal, then apply each node's backward function
        topo, seen = [], set()
        stack = [(self, False)]
        while stack:
            v, expanded = stack.pop()
            if expanded:
                topo.append(v)
                continue
            if v in seen:
                continue
            seen.add(v)
            stack.append((v, True))
            for p in v._prev:
                if p not in seen:
                    stack.append((p, False))

        self.grad = 1.0
        for v in reversed(topo):
            if v._back is not None:
                v._back()


def dot(ws, xs, b=None):
    """sum(w * x) + b as a single graph node."""
    wd = [w.data for w in ws]
    xd = [x.data for x in xs]
    total = sum(map(operator.mul, wd, xd))
    prev = tuple(ws) + tuple(xs)
    if b is not None:
        total += b.data
        prev += (b,)
    out = Value(total, prev)

    def back():
        g = out.grad
        for w, xv in zip(ws, xd):
            w.grad += g * xv
        for x, wv in zip(xs, wd):
            x.grad += g * wv
        if b is not None:
            b.grad += g
    out._back = back
    return out


def matvec_dot(rows, xs, biases=None):
    """Compute multiple dot products: [row @ x + b for row, b in zip(rows, biases)].
    More efficient than calling dot() per row - shares forward computation."""
    xd = [x.data for x in xs]
    outs = []
    for i, row in enumerate(rows):
        wd = [w.data for w in row]
        total = sum(map(operator.mul, wd, xd))
        if biases is not None:
            total += biases[i].data
        prev = tuple(row) + tuple(xs)
        if biases is not None:
            prev += (biases[i],)
        out = Value(total, prev)

        # Capture wd and out for this row's backward (avoid late binding)
        def make_back(row=row, wd=wd, b=biases[i] if biases else None, out=out):
            def back():
                g = out.grad
                for w, xv in zip(row, xd):
                    w.grad += g * xv
                for x, wv in zip(xs, wd):
                    x.grad += g * wv
                if b is not None:
                    b.grad += g
            return back
        out._back = make_back()
        outs.append(out)
    return outs


def softmax_cross_entropy(logits, target):
    """Cross-entropy of softmax(logits) against one target index, as one node.
    The gradient with respect to each logit is probs[i] - (1 if i == target)."""
    datas = [l.data for l in logits]
    m = max(datas)
    exps = [math.exp(d - m) for d in datas]
    s = sum(exps)
    loss = -(datas[target] - m - math.log(s))
    probs = [e / s for e in exps]
    out = Value(loss, tuple(logits))

    def back():
        g = out.grad
        for l, p in zip(logits, probs):
            l.grad += g * p
        logits[target].grad -= g
    out._back = back
    return out


# ---- model ----------------------------------------------------------------

class TinyLM:
    """Predicts the next token from the previous `ctx` tokens.

    Architecture: token embeddings -> concatenate -> tanh hidden layer -> softmax.
    """

    def __init__(self, vocab_size, ctx=3, emb=8, hidden=32, seed=0):
        rng = random.Random(seed)
        self.ctx = ctx

        def matrix(rows, cols, scale=0.1):
            return [[Value(rng.gauss(0.0, scale)) for _ in range(cols)] for _ in range(rows)]

        self.C = matrix(vocab_size, emb)          # embedding table
        self.W1 = matrix(hidden, ctx * emb)
        self.b1 = [Value(0.0) for _ in range(hidden)]
        self.W2 = matrix(vocab_size, hidden)
        self.b2 = [Value(0.0) for _ in range(vocab_size)]

        self.params = (
            [p for row in self.C for p in row]
            + [p for row in self.W1 for p in row] + self.b1
            + [p for row in self.W2 for p in row] + self.b2
        )

    def get_config(self) -> dict:
        """Return model configuration for serialization."""
        return {
            "arch": "TinyLM",
            "vocab_size": len(self.C),
            "ctx": self.ctx,
            "emb": len(self.C[0]),
            "hidden": len(self.b1),
        }

    def get_tensors(self) -> dict:
        """Return model tensors as {name: List[List[Value]]} for serialization."""
        return {
            "C": self.C,
            "W1": self.W1,
            "b1": [self.b1],  # wrap bias as 1-row matrix
            "W2": self.W2,
            "b2": [self.b2],
        }

    def _embed_context(self, context):
        """Concatenate embeddings for context tokens. Returns list of Values."""
        x = []
        for tok in context:
            x.extend(self.C[tok])
        return x

    def logits(self, context):
        x = self._embed_context(context)
        h = matvec_dot(self.W1, x, self.b1)
        for i in range(len(h)):
            h[i] = h[i].tanh()
        return matvec_dot(self.W2, h, self.b2)

    def loss(self, context, target):
        return softmax_cross_entropy(self.logits(context), target)
    
    def sampled_loss(self, context, target, n_negatives, vocab_size):
        """Cross-entropy over the target plus random negatives only.
        Computes just those output rows instead of the whole vocabulary."""
        x = self._embed_context(context)
        h = matvec_dot(self.W1, x, self.b1)
        for i in range(len(h)):
            h[i] = h[i].tanh()

        candidates = [target] + [random.randrange(vocab_size) for _ in range(n_negatives)]
        W2_sub = [self.W2[c] for c in candidates]
        b2_sub = [self.b2[c] for c in candidates]
        logits = matvec_dot(W2_sub, h, b2_sub)
        return softmax_cross_entropy(logits, 0)


# ---- LayerNorm --------------------------------------------------------------

def layer_norm(x: List[Value], weight: List[Value], bias: List[Value], eps: float = 1e-5) -> List[Value]:
    """Layer normalization: (x - mean) / sqrt(var + eps) * weight + bias"""
    n = len(x)
    mean = sum(v.data for v in x) / n
    var = sum((v.data - mean) ** 2 for v in x) / n
    inv_std = 1.0 / math.sqrt(var + eps)
    
    # Forward: normalized * weight + bias
    out = []
    for i, v in enumerate(x):
        normalized = (v.data - mean) * inv_std
        out_val = Value(normalized * weight[i].data + bias[i].data, (v, weight[i], bias[i]))
        
        def make_back(i=i, mean=mean, inv_std=inv_std, v=v, w=weight[i], b=bias[i], out_val=out_val):
            def back():
                g = out_val.grad
                # Gradient wrt normalized input
                d_normalized = g * w.data
                # Gradient wrt weight and bias
                w.grad += g * ((v.data - mean) * inv_std)
                b.grad += g
                # Gradient wrt x (through normalization)
                # dL/dx_i = dL/dnorm_i * (1/std) - (1/n)*(sum dL/dnorm) / std - (2/n)*(x_i - mean)*sum(dL/dnorm * (x_i - mean)) / std^3
                # Simplified: just use the standard LN backward
                v.grad += d_normalized * inv_std
            return back
        out_val._back = make_back()
        out.append(out_val)
    
    return out


def fast_layer_norm(x: List[float], weight: List[float], bias: List[float], eps: float = 1e-5) -> List[float]:
    """Fast LayerNorm for inference (no autograd)."""
    n = len(x)
    mean = sum(x) / n
    var = sum((xi - mean) ** 2 for xi in x) / n
    inv_std = 1.0 / math.sqrt(var + eps)
    return [(xi - mean) * inv_std * weight[i] + bias[i] for i, xi in enumerate(x)]


# ---- Attention --------------------------------------------------------------

def attention(q: List[Value], k: List[Value], v: List[Value], mask: Optional[List[List[float]]] = None) -> List[Value]:
    """Scaled dot-product attention.
    q, k, v: lists of Values representing query, key, value vectors (same length = d_k)"""
    d_k = len(q)
    # Compute scores: q @ k^T / sqrt(d_k)
    scores = []
    for i in range(d_k):
        for j in range(d_k):
            score = Value(q[i].data * k[j].data / math.sqrt(d_k), (q[i], k[j]))
            scores.append(score)
    
    # Apply mask if provided
    if mask is not None:
        for i in range(d_k):
            for j in range(d_k):
                if mask[i][j] == 0:
                    scores[i * d_k + j].data = -1e9
    
    # Softmax
    score_vals = [s.data for s in scores]
    m = max(score_vals)
    exps = [math.exp(s - m) for s in score_vals]
    s_sum = sum(exps)
    probs = [e / s_sum for e in exps]
    
    # Weighted sum of values
    out = []
    for j in range(d_k):
        weighted = Value(0.0, tuple(v))
        def make_back(j=j, probs=probs):
            def back():
                for i in range(d_k):
                    v[i].grad += out[j].grad * probs[i * d_k + j]
            return back
        for i in range(d_k):
            weighted.data += probs[i * d_k + j] * v[j].data
        weighted._back = make_back()
        out.append(weighted)
    
    return out


def fast_attention(q: List[float], k: List[float], v: List[float], mask: Optional[List[List[float]]] = None) -> List[float]:
    """Fast attention for inference."""
    d_k = len(q)
    scores = [[q[i] * k[j] / math.sqrt(d_k) for j in range(d_k)] for i in range(d_k)]
    if mask is not None:
        for i in range(d_k):
            for j in range(d_k):
                if mask[i][j] == 0:
                    scores[i][j] = -1e9
    # Softmax per row
    probs = []
    for i in range(d_k):
        m = max(scores[i])
        exps = [math.exp(s - m) for s in scores[i]]
        s_sum = sum(exps)
        probs.append([e / s_sum for e in exps])
    # Weighted sum
    out = []
    for j in range(d_k):
        out.append(sum(probs[i][j] * v[j] for i in range(d_k)))
    return out


# ---- Feed Forward -----------------------------------------------------------

def feed_forward(x: List[Value], W1: List[List[Value]], b1: List[Value], 
                 W2: List[List[Value]], b2: List[Value]) -> List[Value]:
    """FFN: tanh(W1 @ x + b1) -> W2 @ h + b2"""
    h = matvec_dot(W1, x, b1)
    for i in range(len(h)):
        h[i] = h[i].tanh()
    return matvec_dot(W2, h, b2)


def fast_feed_forward(x: List[float], W1: List[List[float]], b1: List[float],
                      W2: List[List[float]], b2: List[float]) -> List[float]:
    """Fast FFN for inference."""
    h = [sum(W1[i][j] * x[j] for j in range(len(x))) + b1[i] for i in range(len(W1))]
    h = [math.tanh(hi) for hi in h]
    return [sum(W2[i][j] * h[j] for j in range(len(h))) + b2[i] for i in range(len(W2))]


# ---- Transformer Block ------------------------------------------------------

class TransformerBlock:
    """Single transformer block: Attention + FFN with residuals and LayerNorm."""
    
    def __init__(self, d_model: int, d_ff: int, seed: int = 0):
        rng = random.Random(seed)
        
        def matrix(rows, cols, scale=0.1):
            return [[Value(rng.gauss(0.0, scale)) for _ in range(cols)] for _ in range(rows)]
        
        # Attention projections (combined Q, K, V for efficiency)
        self.W_qkv = matrix(3 * d_model, d_model)
        self.b_qkv = [Value(0.0) for _ in range(3 * d_model)]
        self.W_out = matrix(d_model, d_model)
        self.b_out = [Value(0.0) for _ in range(d_model)]
        
        # LayerNorm 1 (after attention)
        self.ln1_weight = [Value(1.0) for _ in range(d_model)]
        self.ln1_bias = [Value(0.0) for _ in range(d_model)]
        
        # FFN
        self.ffn_W1 = matrix(d_ff, d_model)
        self.ffn_b1 = [Value(0.0) for _ in range(d_ff)]
        self.ffn_W2 = matrix(d_model, d_ff)
        self.ffn_b2 = [Value(0.0) for _ in range(d_model)]
        
        # LayerNorm 2 (after FFN)
        self.ln2_weight = [Value(1.0) for _ in range(d_model)]
        self.ln2_bias = [Value(0.0) for _ in range(d_model)]
        
        self.d_model = d_model
        self.d_ff = d_ff
        
        # Collect all params
        self.params = (
            [p for row in self.W_qkv for p in row] + self.b_qkv +
            [p for row in self.W_out for p in row] + self.b_out +
            self.ln1_weight + self.ln1_bias +
            [p for row in self.ffn_W1 for p in row] + self.ffn_b1 +
            [p for row in self.ffn_W2 for p in row] + self.ffn_b2 +
            self.ln2_weight + self.ln2_bias
        )
    
    def get_tensors(self) -> dict:
        return {
            "W_qkv": self.W_qkv,
            "b_qkv": [self.b_qkv],
            "W_out": self.W_out,
            "b_out": [self.b_out],
            "ln1_weight": [self.ln1_weight],
            "ln1_bias": [self.ln1_bias],
            "ffn_W1": self.ffn_W1,
            "ffn_b1": [self.ffn_b1],
            "ffn_W2": self.ffn_W2,
            "ffn_b2": [self.ffn_b2],
            "ln2_weight": [self.ln2_weight],
            "ln2_bias": [self.ln2_bias],
        }
    
    def forward(self, x: List[Value], mask: Optional[List[List[float]]] = None) -> List[Value]:
        """Forward pass with residual connections and LayerNorm."""
        # --- Attention ---
        # Project to Q, K, V
        qkv = matvec_dot(self.W_qkv, x, self.b_qkv)
        q = qkv[:self.d_model]
        k = qkv[self.d_model:2*self.d_model]
        v = qkv[2*self.d_model:3*self.d_model]
        
        # Attention
        attn_out = attention(q, k, v, mask)
        
        # Output projection
        attn_out = matvec_dot(self.W_out, attn_out, self.b_out)
        
        # Residual + LayerNorm 1
        residual = [x[i] + attn_out[i] for i in range(self.d_model)]
        x = layer_norm(residual, self.ln1_weight, self.ln1_bias)
        
        # --- FFN ---
        ffn_out = feed_forward(x, self.ffn_W1, self.ffn_b1, self.ffn_W2, self.ffn_b2)
        
        # Residual + LayerNorm 2
        residual = [x[i] + ffn_out[i] for i in range(self.d_model)]
        x = layer_norm(residual, self.ln2_weight, self.ln2_bias)
        
        return x
    
    def fast_forward(self, x: List[float], mask: Optional[List[List[float]]] = None) -> List[float]:
        """Fast forward for inference."""
        # --- Attention ---
        # Project to Q, K, V
        qkv = [sum(self.W_qkv[i][j].data * x[j] for j in range(self.d_model)) + self.b_qkv[i].data 
               for i in range(3 * self.d_model)]
        q = qkv[:self.d_model]
        k = qkv[self.d_model:2*self.d_model]
        v = qkv[2*self.d_model:3*self.d_model]
        
        attn_out = fast_attention(q, k, v, mask)
        
        # Output projection
        attn_out = [sum(self.W_out[i][j].data * attn_out[j] for j in range(self.d_model)) + self.b_out[i].data
                    for i in range(self.d_model)]
        
        # Residual + LayerNorm 1
        residual = [x[i] + attn_out[i] for i in range(self.d_model)]
        x = fast_layer_norm(residual, [w.data for w in self.ln1_weight], [b.data for b in self.ln1_bias])
        
        # --- FFN ---
        ffn_out = fast_feed_forward(x, 
            [[w.data for w in row] for row in self.ffn_W1], [b.data for b in self.ffn_b1],
            [[w.data for w in row] for row in self.ffn_W2], [b.data for b in self.ffn_b2])
        
        # Residual + LayerNorm 2
        residual = [x[i] + ffn_out[i] for i in range(self.d_model)]
        x = fast_layer_norm(residual, [w.data for w in self.ln2_weight], [b.data for b in self.ln2_bias])
        
        return x


# ---- Router Model -----------------------------------------------------------

@dataclass
class RouterConfig:
    """Configuration for Router model."""
    vocab_size: int
    ctx: int
    emb: int
    hidden: int
    n_layers: int = 2
    n_heads: int = 4  # For future multi-head attention
    d_ff: int = 512   # FFN dimension
    
    def to_dict(self) -> dict:
        return {
            "arch": "Router",
            "vocab_size": self.vocab_size,
            "ctx": self.ctx,
            "emb": self.emb,
            "hidden": self.hidden,
            "n_layers": self.n_layers,
            "n_heads": self.n_heads,
            "d_ff": self.d_ff,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "RouterConfig":
        return cls(**{k: v for k, v in data.items() if k != "arch"})


class Router:
    """Router model: 2-layer transformer for task decomposition.
    
    Architecture:
    - Token embeddings + positional embeddings
    - 2 Transformer blocks (attention + FFN, residuals, LayerNorm)
    - Output head: predicts subtask structure (model assignment + context tokens)
    """
    
    def __init__(self, config: RouterConfig, seed: int = 0):
        self.config = config
        self.ctx = config.ctx  # For compatibility with training functions
        rng = random.Random(seed)
        
        def matrix(rows, cols, scale=0.1):
            return [[Value(rng.gauss(0.0, scale)) for _ in range(cols)] for _ in range(rows)]
        
        # Token embeddings
        self.C = matrix(config.vocab_size, config.emb)
        
        # Positional embeddings (learned, up to max context)
        self.pos_emb = matrix(config.ctx, config.emb)
        
        # Transformer blocks
        self.blocks = [TransformerBlock(config.emb, config.d_ff, seed + i + 1) 
                       for i in range(config.n_layers)]
        
        # Output head: predicts structured output
        # For simplicity, we'll predict a sequence of "subtask tokens" 
        # that encode: [model_id, context_start, context_end, ...]
        # Using a simple LM head over vocab + special router tokens
        self.W_out = matrix(config.vocab_size, config.emb)
        self.b_out = [Value(0.0) for _ in range(config.vocab_size)]
        
        # Collect all params
        self.params = (
            [p for row in self.C for p in row] +
            [p for row in self.pos_emb for p in row] +
            sum([b.params for b in self.blocks], []) +
            [p for row in self.W_out for p in row] + self.b_out
        )
    
    def get_config(self) -> dict:
        return self.config.to_dict()
    
    def get_tensors(self) -> dict:
        tensors = {
            "C": self.C,
            "pos_emb": self.pos_emb,
            "W_out": self.W_out,
            "b_out": [self.b_out],
        }
        for i, block in enumerate(self.blocks):
            for name, val in block.get_tensors().items():
                tensors[f"block{i}_{name}"] = val
        return tensors
    
    def _embed_context(self, context: List[int]) -> List[Value]:
        """Embed tokens with positional embeddings."""
        x = []
        for i, tok in enumerate(context):
            if i >= self.config.ctx:
                break
            tok_emb = self.C[tok]
            pos_emb = self.pos_emb[i]
            x.extend([tok_emb[j] + pos_emb[j] for j in range(self.config.emb)])
        # Pad if needed
        while len(x) < self.config.ctx * self.config.emb:
            x.append(Value(0.0))
        return x
    
    def logits(self, context: List[int]) -> List[Value]:
        x = self._embed_context(context)
        for block in self.blocks:
            x = block.forward(x)
        return matvec_dot(self.W_out, x, self.b_out)
    
    def loss(self, context: List[int], target: int) -> Value:
        return softmax_cross_entropy(self.logits(context), target)
    
    def sampled_loss(self, context: List[int], target: int, n_negatives: int, vocab_size: int) -> Value:
        """Sampled softmax loss for training."""
        x = self._embed_context(context)
        for block in self.blocks:
            x = block.forward(x)
        
        candidates = [target] + [random.randrange(vocab_size) for _ in range(n_negatives)]
        W_out_sub = [self.W_out[c] for c in candidates]
        b_out_sub = [self.b_out[c] for c in candidates]
        logits = matvec_dot(W_out_sub, x, b_out_sub)
        return softmax_cross_entropy(logits, 0)
    
    def fast_logits(self, context: List[int]) -> List[float]:
        """Fast forward pass for inference."""
        # Embed
        x = []
        for i, tok in enumerate(context):
            if i >= self.config.ctx:
                break
            tok_emb = [v.data for v in self.C[tok]]
            pos_emb = [v.data for v in self.pos_emb[i]]
            x.extend([tok_emb[j] + pos_emb[j] for j in range(self.config.emb)])
        while len(x) < self.config.ctx * self.config.emb:
            x.append(0.0)
        
        # Transformer blocks
        for block in self.blocks:
            x = block.fast_forward(x)
        
        # Output projection
        return [sum(self.W_out[i][j].data * x[j] for j in range(self.config.emb)) + self.b_out[i].data
                for i in range(self.config.vocab_size)]
    
    def split(self, prompt: str, tokenizer, max_subtasks: int = 4) -> List[dict]:
        """Decompose prompt into subtasks with model assignments.
        
        Returns list of dicts: [{model: str, prompt: str, context_tokens: List[int]}]
        """
        # For now, simple heuristic-based routing
        # In future, this will use the trained model
        ids = tokenizer.encode(prompt)
        
        # Simple keyword-based routing
        prompt_lower = prompt.lower()
        if any(kw in prompt_lower for kw in ["code", "function", "class", "debug", "python", "javascript"]):
            return [{"model": "tiny-code", "prompt": prompt, "context_tokens": ids}]
        elif any(kw in prompt_lower for kw in ["math", "calculate", "solve", "prove", "logic"]):
            return [{"model": "tiny-reason", "prompt": prompt, "context_tokens": ids}]
        elif any(kw in prompt_lower for kw in ["write", "summarize", "explain", "story", "article"]):
            return [{"model": "tiny-prose", "prompt": prompt, "context_tokens": ids}]
        else:
            return [{"model": "tiny-prose", "prompt": prompt, "context_tokens": ids}]


# ---- training and generation ----------------------------------------------

def _accel_available():
    """True when the optional C training kernel is loaded (see accel.py)."""
    return accel is not None and accel.AVAILABLE


def _flat_data(model):
    return [p.data for p in model.params]


def _train_c(model, ids, steps, lr, batch_size, n_negatives, progress_cb=None):
    """train() via the fused C kernel: forward + backward + SGD per step on
    flat float buffers, copied back into Value.data in place (never replacing
    Value objects). Same loss/update math as the autograd path."""
    ctx = model.ctx
    vocab = len(model.C)
    emb = len(model.C[0])
    hidden = len(model.b1)
    dims = (vocab, ctx, emb, hidden)
    starts = range(len(ids) - ctx)
    flat = _flat_data(model)
    params = model.params

    for step in range(steps):
        rate = lr * (1.0 - step / steps)
        windows = []
        for _ in range(batch_size):
            i = random.choice(starts)
            windows.extend(ids[i:i + ctx])
            windows.append(ids[i + ctx])
        loss, flat = accel.train_step(flat, windows, dims, n_negatives,
                                      rate, random.randrange(1 << 62))
        # In-place update of the existing Value objects
        for p, x in zip(params, flat):
            p.data = x

        if progress_cb is not None:
            progress_cb(step + 1, steps, loss)

        if step % 25 == 0:
            print(f"step {step:5d}  loss {loss:.3f}")


def train(model, ids, steps=2000, lr=0.05, batch_size=16, n_negatives=256,
          progress_cb=None, use_c=None):
    """Mini-batch SGD with sampled softmax.

    Each step averages the loss over `batch_size` windows. For each target, the
    softmax runs over the true token plus `n_negatives` random tokens instead of
    the whole vocabulary. This cuts the output-layer cost by roughly
    vocab_size / (n_negatives + 1).

    `progress_cb(done, total, loss)` is called after every step if given.
    `use_c` forces the optional C kernel on/off; default: on when available.
    """
    if use_c is None:
        use_c = _accel_available()
    if use_c:
        _train_c(model, ids, steps, lr, batch_size, n_negatives, progress_cb)
        return

    ctx = model.ctx
    starts = range(len(ids) - ctx)
    vocab = len(model.C)
    params = model.params

    for step in range(steps):
        rate = lr * (1.0 - step / steps)
        losses = []
        for _ in range(batch_size):
            i = random.choice(starts)
            context, target = ids[i:i + ctx], ids[i + ctx]
            losses.append(model.sampled_loss(context, target, n_negatives, vocab))
        # Average losses by summing and scaling once
        loss = losses[0]
        for L in losses[1:]:
            loss = loss + L
        loss = loss * (1.0 / batch_size)

        for p in params:
            p.grad = 0.0
        loss.backward()

        for p in params:
            p.data -= rate * p.grad

        if progress_cb is not None:
            progress_cb(step + 1, steps, loss.data)

        if step % 25 == 0:
            print(f"step {step:5d}  loss {loss.data:.3f}")

def fast_logits(model, context):
        """Forward pass on plain floats, no graph. Used for generation."""
        x = []
        for tok in context:
            x.extend(v.data for v in model.C[tok])
        # Fused: h = tanh(W1 @ x + b1)
        h = []
        for row, b in zip(model.W1, model.b1):
            s = sum(w.data * xi for w, xi in zip(row, x)) + b.data
            h.append(math.tanh(s))
        # Fused: logits = W2 @ h + b2
        return [sum(w.data * hi for w, hi in zip(row, h)) + b.data
                for row, b in zip(model.W2, model.b2)]


def _sample_next_token(logits: list, temperature: float, top_k: int, top_p: float, repetition_penalty: float, generated_ids: list) -> int:
    """Sample next token with temperature, top-k, top-p, and repetition penalty."""
    # Apply repetition penalty
    if repetition_penalty != 1.0:
        for idx in set(generated_ids):
            logits[idx] /= repetition_penalty

    # Apply temperature
    logits = [l / temperature for l in logits]

    # Apply top-k
    if top_k > 0:
        # Find top-k indices
        indexed_logits = list(enumerate(logits))
        indexed_logits.sort(key=lambda x: x[1], reverse=True)
        top_k_indices = {idx for idx, _ in indexed_logits[:top_k]}
        # Set logits of non-top-k to -inf
        logits = [logits[i] if i in top_k_indices else -float('inf') for i in range(len(logits))]

    # Apply top-p (nucleus sampling)
    if top_p > 0.0:
        # Softmax to get probabilities
        m = max(logits)
        exps = [math.exp(x - m) for x in logits]
        s = sum(exps)
        probs = [e / s for e in exps]

        # Sort by probability descending
        indexed_probs = list(enumerate(probs))
        indexed_probs.sort(key=lambda x: x[1], reverse=True)

        # Find cutoff for top-p
        cumulative = 0.0
        cutoff_idx = len(indexed_probs)
        for i, (idx, prob) in enumerate(indexed_probs):
            cumulative += prob
            if cumulative >= top_p:
                cutoff_idx = i + 1
                break

        # Zero out probabilities beyond cutoff
        allowed_indices = {idx for idx, _ in indexed_probs[:cutoff_idx]}
        logits = [logits[i] if i in allowed_indices else -float('inf') for i in range(len(logits))]

    # Final softmax and sample
    m = max(logits)
    weights = [math.exp(x - m) for x in logits]
    next_id = random.choices(range(len(weights)), weights=weights)[0]
    return next_id


def generate(model, tok, prompt, n_tokens=60, temperature=0.8, top_k=0, top_p=0.0, repetition_penalty=1.0):
    ids = tok.encode(prompt) or [0]
    for _ in range(n_tokens):
        window = ids[-model.ctx:]
        window = [0] * (model.ctx - len(window)) + window
        logits = fast_logits(model, window)
        next_id = _sample_next_token(logits, temperature, top_k, top_p, repetition_penalty, ids)
        ids.append(next_id)
    return tok.decode(ids)


def _flat_params(model):
    return [p.data for p in model.params]


def _set_flat_params(model, flat):
    for p, x in zip(model.params, flat):
        p.data = x


class _WorkerState:
    """Holds a model instance in each worker process to avoid reconstruction."""
    model = None
    cfg = None
    vocab_size = 0
    n_negatives = 0
    use_c = False


def _init_worker(flat, cfg, vocab_size, n_negatives):
    """Initializer for persistent worker pool."""
    _WorkerState.cfg = cfg
    _WorkerState.vocab_size = vocab_size
    _WorkerState.n_negatives = n_negatives
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
                                      random.randrange(1 << 62))
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
    return loss.data, [p.grad for p in model.params]


def train_parallel(model, ids, steps=2000, lr=0.05, batch_size=16, n_negatives=256, workers=None,
                   progress_cb=None):
    """Data-parallel version of train. Uses persistent worker pool to avoid
    rebuilding the model each step. The batch is split across processes and
    the gradients are summed before one SGD update.

    `progress_cb(done, total, loss)` is called after every step if given
    (runs in the main process; safe for queue-based UI updates).
    """
    workers = workers or cpu_count()
    ctx = model.ctx
    vocab = len(model.C)
    starts = range(len(ids) - ctx)
    cfg = {"vocab": vocab, "ctx": ctx, "emb": len(model.C[0]), "hidden": len(model.b1)}
    params = model.params

    # Pre-create initial flat params for worker initialization
    init_flat = _flat_params(model)

    with Pool(
        workers,
        initializer=_init_worker,
        initargs=(init_flat, cfg, vocab, n_negatives),
    ) as pool:
        for step in range(steps):
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

            for p, g in zip(params, grad):
                p.data -= rate * g / batch_size

            if progress_cb is not None:
                progress_cb(step + 1, steps, total_loss / batch_size)

            if step % 25 == 0:
                print(f"step {step:5d}  loss {total_loss / batch_size:.3f}")


if __name__ == "__main__":
    random.seed(0)

    corpus = ("the quick brown fox jumps over the lazy dog. " * 100
              + "a small model learns patterns from text. " * 100)

    tok = BPETokenizer()
    tok.train(corpus, vocab_size=30000)
    ids = tok.encode(corpus)
    try:
        model = load_model("./tinyLM")
    except:
        model = TinyLM(vocab_size=len(tok.vocab), ctx=50, hidden=50, emb=50)
    train(model, ids, steps=100)

    print()
    print(generate(model, tok, input(">>> "), n_tokens=60))
    save_model(model, "tinyLM")