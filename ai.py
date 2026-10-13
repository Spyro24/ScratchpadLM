import math
import random
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

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
    
    def split(self, prompt: str, tokenizer, max_subtasks: int = 4, catalog=None) -> List[dict]:
        """Decompose prompt into subtasks with model assignments.
        
        Returns list of dicts: [{model: str, prompt: str, context_tokens: List[int]}]
        
        Args:
            prompt: Input prompt to route
            tokenizer: BPETokenizer for encoding
            max_subtasks: Maximum number of subtasks to return
            catalog: Optional ModelCatalog to use domain_keywords for routing
        """
        ids = tokenizer.encode(prompt)
        prompt_lower = prompt.lower()
        
        # If catalog provided, use domain_keywords for routing
        if catalog is not None:
            # Score each model by keyword matches
            scores = {}
            for spec in catalog.list():
                if spec.name == "router":
                    continue  # Skip router itself
                score = sum(1 for kw in spec.domain_keywords if kw in prompt_lower)
                if score > 0:
                    scores[spec.name] = score
            
            if scores:
                # Return top-scoring models (up to max_subtasks)
                sorted_models = sorted(scores.items(), key=lambda x: x[1], reverse=True)
                return [
                    {"model": model_name, "prompt": prompt, "context_tokens": ids}
                    for model_name, _ in sorted_models[:max_subtasks]
                ]
        
        # Fallback to hardcoded keyword-based routing
        if any(kw in prompt_lower for kw in ["code", "function", "class", "debug", "python", "javascript"]):
            return [{"model": "tiny-code", "prompt": prompt, "context_tokens": ids}]
        elif any(kw in prompt_lower for kw in ["math", "calculate", "solve", "prove", "logic"]):
            return [{"model": "tiny-reason", "prompt": prompt, "context_tokens": ids}]
        elif any(kw in prompt_lower for kw in ["write", "summarize", "explain", "story", "article"]):
            return [{"model": "tiny-prose", "prompt": prompt, "context_tokens": ids}]
        else:
            return [{"model": "tiny-prose", "prompt": prompt, "context_tokens": ids}]


# ---- Specialized Model Base (≤100K params) ------------------------------------

@dataclass
class SpecializedConfig:
    """Configuration for a specialized model (≤100K params)."""
    vocab_size: int
    ctx: int
    emb: int
    hidden: int
    n_layers: int = 1
    d_ff: int = 256   # FFN dimension (typically 4*emb for standard, but we use smaller for param budget)
    model_type: str = "specialized"  # e.g., "code", "reason", "prose", "embed"
    
    def to_dict(self) -> dict:
        return {
            "arch": "SpecializedLM",
            "vocab_size": self.vocab_size,
            "ctx": self.ctx,
            "emb": self.emb,
            "hidden": self.hidden,
            "n_layers": self.n_layers,
            "d_ff": self.d_ff,
            "model_type": self.model_type,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "SpecializedConfig":
        return cls(**{k: v for k, v in data.items() if k != "arch"})
    
    def estimate_params(self) -> int:
        """Estimate parameter count for this config."""
        # Embeddings: vocab_size * emb
        # Positional: ctx * emb
        # Per block: attention (4*emb*emb + 4*emb) + ln1 (2*emb) + ffn (2*emb*d_ff + 2*d_ff) + ln2 (2*emb)
        # Output head: vocab_size * emb + vocab_size
        emb, ctx, vocab = self.emb, self.ctx, self.vocab_size
        d_ff = self.d_ff
        n_layers = self.n_layers
        
        embed_params = vocab * emb + ctx * emb
        block_params = n_layers * (4 * emb * emb + 4 * emb + 2 * emb + 2 * emb * d_ff + 2 * d_ff + 2 * emb)
        output_params = vocab * emb + vocab
        return embed_params + block_params + output_params


class SpecializedLM:
    """Specialized language model: 1-layer transformer (≤100K params).
    
    Architecture:
    - Token embeddings + positional embeddings
    - 1 Transformer block (attention + FFN, residuals, LayerNorm)
    - Output head: standard LM head over vocabulary
    
    Designed to fit within 100K parameter budget while maintaining quality.
    """
    
    def __init__(self, config: SpecializedConfig, seed: int = 0):
        self.config = config
        self.ctx = config.ctx  # For compatibility with training functions
        rng = random.Random(seed)
        
        def matrix(rows, cols, scale=0.1):
            return [[Value(rng.gauss(0.0, scale)) for _ in range(cols)] for _ in range(rows)]
        
        # Token embeddings
        self.C = matrix(config.vocab_size, config.emb)
        
        # Positional embeddings (learned, up to max context)
        self.pos_emb = matrix(config.ctx, config.emb)
        
        # Transformer blocks (typically 1 for specialized models)
        self.blocks = [TransformerBlock(config.emb, config.d_ff, seed + i + 1) 
                       for i in range(config.n_layers)]
        
        # Output head: standard LM head
        self.W_out = matrix(config.vocab_size, config.emb)
        self.b_out = [Value(0.0) for _ in range(config.vocab_size)]
        
        # Collect all params
        self.params = (
            [p for row in self.C for p in row] +
            [p for row in self.pos_emb for p in row] +
            sum([b.params for b in self.blocks], []) +
            [p for row in self.W_out for p in row] + self.b_out
        )
        
        # Enforce param limit
        from ai import count_params, enforce_param_limit
        n_params = count_params(self)
        enforce_param_limit(self, 100_000, f"SpecializedLM-{config.model_type}")
    
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
        """Fast forward pass for inference (no autograd)."""
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


def create_specialized_model(config: SpecializedConfig, seed: int = 0) -> SpecializedLM:
    """Factory function to create a specialized model from config."""
    return SpecializedLM(config, seed)


# Preset configs for common specialized model types
def get_specialized_presets() -> Dict[str, SpecializedConfig]:
    """Get preset configurations for common specialized model types.
    
    All presets are designed to stay under 100K parameters.
    Using emb=32, d_ff=128 to fit within budget.
    """
    base_kwargs = dict(vocab_size=1000, ctx=128, emb=32, hidden=32, n_layers=1, d_ff=128)
    return {
        "code": SpecializedConfig(model_type="code", **base_kwargs),
        "reason": SpecializedConfig(model_type="reason", **base_kwargs),
        "prose": SpecializedConfig(model_type="prose", **base_kwargs),
        "embed": SpecializedConfig(model_type="embed", **base_kwargs),
        "sql": SpecializedConfig(model_type="sql", **base_kwargs),
        "docs": SpecializedConfig(model_type="docs", **base_kwargs),
        "shell": SpecializedConfig(model_type="shell", **base_kwargs),
        "data": SpecializedConfig(model_type="data", **base_kwargs),
    }


# ---- Router Training Data Format --------------------------------------------

# Model name to slot mapping (fixed assignment for known models)
_MODEL_SLOTS = {
    "tiny-code": 0,
    "tiny-reason": 1,
    "tiny-prose": 2,
    "tiny-embed": 3,
    "tiny-sql": 4,
    "tiny-docs": 5,
    "tiny-shell": 6,
    "tiny-data": 7,
}


def _stable_hash(s: str, mod: int) -> int:
    """Stable hash function (not randomized across runs)."""
    h = 0
    for ch in s:
        h = (h * 1315423911 + ord(ch)) % mod
    return h


def _model_slot(name: str) -> int:
    """Get stable slot index for a model name."""
    return _MODEL_SLOTS.get(name, _stable_hash(name, 8))


@dataclass
class RouterTrainingExample:
    """A single training example for the router: (prompt, subtasks).
    
    Subtasks are a list of dicts with keys:
      - model: target model name (e.g., "tiny-code")
      - prompt: the sub-prompt for that model
      - context_tokens: optional list of token IDs for context
    """
    prompt: str
    subtasks: List[dict]


def _ensure_vocab_size(tokenizer, min_vocab: int) -> int:
    """Extend tokenizer to have at least min_vocab tokens.
    
    Returns the actual vocab size.
    """
    actual = len(tokenizer.vocab)
    if actual >= min_vocab:
        return actual
    # Create repetitive text with limited alphabet to maximize merge creation
    # Use a rotating pattern of bytes that creates many frequent pairs
    import itertools
    chars = [chr(i) for i in range(97, 110)]  # a-n (14 chars)
    # Generate all 2-char combinations
    pairs = [a + b for a, b in itertools.product(chars, repeat=2)]
    dummy = " ".join(p * 100 for p in pairs)
    tokenizer.extend(dummy, vocab_size=min_vocab + 20)
    return len(tokenizer.vocab)


def encode_router_example(example: RouterTrainingExample, tokenizer, max_ctx: int, vocab_size: int) -> List[int]:
    """Encode a router training example into a flat token sequence.
    
    Format: [prompt_tokens, SEP, subtask1_model, subtask1_prompt_tokens, SEP, ...]
    
    Special tokens (reserved at top of vocab):
      - SEP = vocab_size - 1
      - Model tokens: vocab_size - 10 .. vocab_size - 3 (8 model slots)
    """
    # Ensure tokenizer has enough vocab
    actual_vocab = _ensure_vocab_size(tokenizer, vocab_size)
    
    # Special token IDs (use actual vocab size)
    SEP = actual_vocab - 1
    MODEL_BASE = actual_vocab - 10  # 8 slots: -10 to -3
    
    def model_token(name: str) -> int:
        # Stable slot for model name
        slot = _model_slot(name)
        return MODEL_BASE + slot
    
    tokens = []
    # Encode main prompt
    prompt_ids = tokenizer.encode(example.prompt)
    tokens.extend(prompt_ids[-max_ctx:])
    tokens.append(SEP)
    
    # Encode each subtask
    for subtask in example.subtasks:
        model_name = subtask.get("model", "tiny-prose")
        tokens.append(model_token(model_name))
        
        sub_prompt = subtask.get("prompt", example.prompt)
        sub_ids = tokenizer.encode(sub_prompt)
        tokens.extend(sub_ids[-max_ctx:])
        tokens.append(SEP)
    
    return tokens


def decode_router_sequence(tokens: List[int], tokenizer, vocab_size: int) -> RouterTrainingExample:
    """Decode a flat token sequence back into a RouterTrainingExample."""
    actual_vocab = len(tokenizer.vocab)
    SEP = actual_vocab - 1
    MODEL_BASE = actual_vocab - 10
    
    # Split by SEP
    parts = []
    current = []
    for t in tokens:
        if t == SEP:
            parts.append(current)
            current = []
        else:
            current.append(t)
    if current:
        parts.append(current)
    
    if not parts:
        return RouterTrainingExample("", [])
    
    prompt = tokenizer.decode(parts[0])
    subtasks = []
    
    # Model name mapping (must match encode)
    model_names = ["tiny-code", "tiny-reason", "tiny-prose", "tiny-embed",
                   "tiny-sql", "tiny-docs", "tiny-shell", "tiny-data"]
    
    for part in parts[1:]:
        if not part:
            continue
        first = part[0]
        if first >= MODEL_BASE:
            model_idx = first - MODEL_BASE
            model_name = model_names[model_idx % len(model_names)]
        else:
            model_name = "tiny-prose"
        
        sub_prompt = tokenizer.decode(part[1:]) if len(part) > 1 else prompt
        subtasks.append({"model": model_name, "prompt": sub_prompt})
    
    return RouterTrainingExample(prompt, subtasks)


def create_router_training_data(tokenizer, vocab_size: int, ctx: int,
                                 examples: List[RouterTrainingExample]) -> List[int]:
    """Create a flat token ID sequence for router training from examples.
    
    Concatenates encoded examples with SEP tokens between them.
    """
    _ensure_vocab_size(tokenizer, vocab_size)
    actual_vocab = len(tokenizer.vocab)
    SEP = actual_vocab - 1
    
    all_tokens = []
    for ex in examples:
        encoded = encode_router_example(ex, tokenizer, ctx, vocab_size)
        all_tokens.extend(encoded)
        all_tokens.append(SEP)
    return all_tokens


# ---- Router Pretraining -----------------------------------------------------

def train_router(router, tokenizer, training_examples: List[RouterTrainingExample],
                 steps: int = 10000, lr: float = 0.05, batch_size: int = 16,
                 n_negatives: int = 256, progress_cb=None, **train_kwargs):
    """Train a router model on (prompt, subtasks) pairs.
    
    Args:
        router: Router model instance
        tokenizer: BPETokenizer for encoding
        training_examples: List of RouterTrainingExample objects
        steps: Number of training steps
        lr: Learning rate
        batch_size: Batch size per step
        n_negatives: Number of negative samples for sampled softmax
        progress_cb: Optional callback(done, total, loss, val_loss=None, lr=None)
        **train_kwargs: Additional arguments passed to train()
    
    Returns:
        The trained router model (modified in place)
    """
    # Create flat token sequence from examples
    vocab_size = len(router.C)
    ctx = router.ctx
    ids = create_router_training_data(tokenizer, vocab_size, ctx, training_examples)
    
    if len(ids) < ctx + 1:
        raise ValueError(f"Not enough training data: {len(ids)} tokens, need at least {ctx + 1}")
    
    # Train using the standard train function
    # The router's sampled_loss method is compatible with train()
    train(router, ids, steps=steps, lr=lr, batch_size=batch_size,
          n_negatives=n_negatives, progress_cb=progress_cb, **train_kwargs)
    
    return router


def generate_router_training_data(corpus: str, tokenizer, vocab_size: int, ctx: int,
                                   domain_keywords: Optional[dict] = None) -> List[RouterTrainingExample]:
    """Generate router training examples from a corpus using heuristic domain detection.
    
    Args:
        corpus: Text corpus to generate examples from
        tokenizer: BPETokenizer for encoding
        vocab_size: Vocabulary size (for special token allocation)
        ctx: Context window size
        domain_keywords: Optional dict mapping model names to keyword lists
                        e.g., {"tiny-code": ["code", "function", "python"],
                               "tiny-reason": ["math", "calculate", "logic"]}
    
    Returns:
        List of RouterTrainingExample objects
    """
    if domain_keywords is None:
        domain_keywords = {
            "tiny-code": ["code", "function", "class", "debug", "python", "javascript", "def ", "import ", "return "],
            "tiny-reason": ["math", "calculate", "solve", "prove", "logic", "theorem", "equation"],
            "tiny-prose": ["write", "summarize", "explain", "story", "article", "describe", "tell me"],
        }
    
    # Split corpus into sentences/paragraphs
    import re
    sentences = re.split(r'[.!?]\s+', corpus)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
    
    examples = []
    for sent in sentences:
        sent_lower = sent.lower()
        # Detect domain
        best_model = "tiny-prose"
        best_score = 0
        for model, keywords in domain_keywords.items():
            score = sum(1 for kw in keywords if kw in sent_lower)
            if score > best_score:
                best_score = score
                best_model = model
        
        # Create example with single subtask (can be extended to multiple)
        examples.append(RouterTrainingExample(
            prompt=sent,
            subtasks=[{"model": best_model, "prompt": sent}]
        ))
    
    return examples


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
             auto_patience: int = 10, auto_clip_relax: bool = True):
    """train() via the fused C kernel with gradient clipping support.
    
    C kernel supports: LR per step, gradient clipping.
    Python handles: LR scheduling, validation, early stopping, checkpointing,
    automatic adjustment (plateau LR reduction, clip relaxation).
    """
    ctx = model.ctx
    vocab = len(model.C)
    emb = len(model.C[0])
    hidden = len(model.b1)
    dims = (vocab, ctx, emb, hidden)
    starts = range(len(ids) - ctx)
    flat = _flat_data(model)
    params = model.params

    # Setup LR scheduler
    from ai import LR_Scheduler
    scheduler = LR_Scheduler(lr, steps, lr_scheduler, warmup_steps, min_lr)

    # Setup early stopping
    from ai import EarlyStopping
    early_stopper = EarlyStopping(patience=patience, min_delta=min_delta) if early_stopping else None

    # Setup automatic adjustment
    from ai import AutoAdjust
    adjuster = AutoAdjust(
        enabled=auto_adjust, factor=auto_factor, patience=auto_patience,
        min_delta=min_delta, warmup_ignore=warmup_steps,
        max_grad_clip=grad_clip if (auto_adjust and auto_clip_relax) else 0.0,
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
        starts = range(len(train_ids) - ctx)
    else:
        train_ids = ids

    for step in range(steps):
        rate = scheduler.get_lr(step) * adjuster.multiplier
        windows = []
        for _ in range(batch_size):
            i = random.choice(list(starts))
            windows.extend(train_ids[i:i + ctx])
            windows.append(train_ids[i + ctx])
        
        # C kernel with gradient clipping
        loss, flat = accel.train_step(flat, windows, dims, n_negatives,
                                      rate, random.randrange(1 << 62),
                                      max_norm=adjuster.grad_clip_now())
        # In-place update of the existing Value objects
        for p, x in zip(params, flat):
            p.data = x

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
            from ai import Checkpoint, save_checkpoint, _flat_params
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

    # Save final checkpoint
    if checkpoint_every > 0:
        from ai import Checkpoint, save_checkpoint, _flat_params
        ckpt = Checkpoint(
            step=steps,
            model_state=_flat_params(model),
            config=model.get_config(),
            loss=loss,
            val_loss=val_loss,
            lr=scheduler.get_lr(steps - 1),
        )
        ckpt_file = checkpoint_path / "checkpoint_final.json"
        save_checkpoint(ckpt, str(ckpt_file))


def train(model, ids, steps=2000, lr=0.05, batch_size=16, n_negatives=256,
          progress_cb=None, use_c=None,
          lr_scheduler: str = "cosine_warmup", warmup_steps: int = 100, min_lr: float = 0.0,
          grad_clip: float = 1.0, val_ids=None, val_split: float = 0.0, eval_every: int = 50,
          early_stopping: bool = True, patience: int = 5, min_delta: float = 1e-4,
          checkpoint_every: int = 100, checkpoint_dir: str = "checkpoints",
          auto_adjust: bool = True, auto_factor: float = 0.5,
          auto_patience: int = 10, auto_clip_relax: bool = True):
    """Mini-batch SGD with sampled softmax and advanced training features.

    Each step averages the loss over `batch_size` windows. For each target, the
    softmax runs over the true token plus `n_negatives` random tokens instead of
    the whole vocabulary. This cuts the output-layer cost by roughly
    vocab_size / (n_negatives + 1).

    New features:
    - LR scheduling: cosine_warmup, linear, constant
    - Gradient clipping by global norm
    - Validation evaluation with perplexity
    - Early stopping with best model restoration
    - Checkpointing for resume capability
    - Automatic adjustment: LR reduction on plateau + auto grad-clip relaxation

    `progress_cb(done, total, loss, val_loss=None, lr=None)` is called after every step if given.
    `use_c` forces the optional C kernel on/off; default: on when available.
    Note: C kernel supports LR scheduling and gradient clipping; Python handles validation, early stopping, checkpointing.
    """
    # Disable C kernel when advanced features are requested that C doesn't support
    # C kernel now supports: gradient clipping, LR per step (for scheduling)
    # C kernel does NOT support: validation, early stopping, checkpointing
    advanced_features = (
        val_ids is not None or
        val_split > 0 or
        eval_every > 0 or
        early_stopping or
        checkpoint_every > 0
    )
    
    if use_c is None:
        use_c = _accel_available() and not advanced_features
    if use_c:
        # C kernel path with gradient clipping support
        _train_c(model, ids, steps, lr, batch_size, n_negatives, progress_cb,
                 lr_scheduler=lr_scheduler, warmup_steps=warmup_steps, min_lr=min_lr,
                 grad_clip=grad_clip, val_ids=val_ids, val_split=val_split, eval_every=eval_every,
                 early_stopping=early_stopping, patience=patience, min_delta=min_delta,
                 checkpoint_every=checkpoint_every, checkpoint_dir=checkpoint_dir,
                 auto_adjust=auto_adjust, auto_factor=auto_factor,
                 auto_patience=auto_patience, auto_clip_relax=auto_clip_relax)
        return

    ctx = model.ctx
    starts = range(len(ids) - ctx)
    vocab = len(model.C)
    params = model.params

    # Split validation data if provided as ratio
    if val_ids is None and val_split > 0:
        split_idx = int(len(ids) * (1 - val_split))
        train_ids = ids[:split_idx]
        val_ids = ids[split_idx:]
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
        max_grad_clip=grad_clip if (auto_adjust and auto_clip_relax) else 0.0,
    )

    # Setup checkpoint directory
    from pathlib import Path
    checkpoint_path = Path(checkpoint_dir)
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    for step in range(steps):
        rate = scheduler.get_lr(step) * adjuster.multiplier
        losses = []
        for _ in range(batch_size):
            i = random.choice(starts)
            context, target = train_ids[i:i + ctx], train_ids[i + ctx]
            losses.append(model.sampled_loss(context, target, n_negatives, vocab))
        # Average losses by summing and scaling once
        loss = losses[0]
        for L in losses[1:]:
            loss = loss + L
        loss = loss * (1.0 / batch_size)

        for p in params:
            p.grad = 0.0
        loss.backward()

        # Gradient clipping (with automatic relaxation if chronically clipping)
        clip_now = adjuster.grad_clip_now()
        if clip_now > 0:
            norm = clip_grads(params, clip_now)
            adjuster.note_grad_clip(norm)

        for p in params:
            p.data -= rate * p.grad

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

    # Save final checkpoint
    if checkpoint_every > 0:
        ckpt = Checkpoint(
            step=steps,
            model_state=_flat_params(model),
            config=model.get_config(),
            loss=loss.data,
            val_loss=val_loss,
            lr=scheduler.get_lr(steps - 1),
        )
        ckpt_file = checkpoint_path / "checkpoint_final.json"
        save_checkpoint(ckpt, str(ckpt_file))

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
    # Use model's own fast_logits if available (for transformer models), else use legacy
    model_fast_logits = getattr(model, 'fast_logits', None)
    for _ in range(n_tokens):
        window = ids[-model.ctx:]
        window = [0] * (model.ctx - len(window)) + window
        if model_fast_logits is not None:
            logits = model_fast_logits(window)
        else:
            logits = fast_logits(model, window)
        next_id = _sample_next_token(logits, temperature, top_k, top_p, repetition_penalty, ids)
        ids.append(next_id)
    return tok.decode(ids)


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

@dataclass
class Checkpoint:
    """Training checkpoint with model state, optimizer state, and metadata."""
    step: int
    model_state: list  # flat list of parameter values
    config: dict
    loss: float
    val_loss: Optional[float] = None
    lr: float = 0.0

    def to_dict(self) -> dict:
        import json
        return {
            "step": self.step,
            "model_state": self.model_state,
            "config": self.config,
            "loss": self.loss,
            "val_loss": self.val_loss,
            "lr": self.lr,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Checkpoint":
        return cls(
            step=data["step"],
            model_state=data["model_state"],
            config=data["config"],
            loss=data["loss"],
            val_loss=data.get("val_loss"),
            lr=data.get("lr", 0.0),
        )


def save_checkpoint(checkpoint: Checkpoint, path: str):
    """Save checkpoint to JSON file."""
    import json
    from pathlib import Path
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(checkpoint.to_dict(), f)


def load_checkpoint(path: str) -> Checkpoint:
    """Load checkpoint from JSON file."""
    import json
    with open(path, "r") as f:
        data = json.load(f)
    return Checkpoint.from_dict(data)


def resume_from_checkpoint(model, checkpoint: Checkpoint):
    """Restore model from checkpoint."""
    for p, val in zip(model.params, checkpoint.model_state):
        p.data = val


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
    max_norm = 0.0


def _init_worker(flat, cfg, vocab_size, n_negatives, max_norm=0.0):
    """Initializer for persistent worker pool."""
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
        from ai import clip_grads
        clip_grads(model.params, _WorkerState.max_norm)
    return loss.data, [p.grad for p in model.params]


def train_parallel(model, ids, steps=2000, lr=0.05, batch_size=16, n_negatives=256, workers=None,
                   progress_cb=None, grad_clip: float = 0.0):
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
        initargs=(init_flat, cfg, vocab, n_negatives, grad_clip),
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

            # Apply gradient clipping on main process (after summing)
            if grad_clip > 0:
                import math
                total_norm = math.sqrt(sum(g * g for g in grad))
                if total_norm > grad_clip and total_norm > 0:
                    scale = grad_clip / total_norm
                    for k in range(len(grad)):
                        grad[k] *= scale

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