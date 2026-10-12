import math
import random

from tokenizer import BPETokenizer

import struct
from multiprocessing import Pool, cpu_count

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

    Layout (little-endian):
      magic "TLM4"
      4 x uint32: vocab_size, ctx, emb, hidden
      for each matrix (C, W1, W2) and bias vector (b1, b2):
        uint32 rows, uint32 cols
        float32 scale per row
        packed 4-bit values, (rows * cols + 1) // 2 bytes
    """
    vocab_size = len(model.C)
    emb = len(model.C[0])
    hidden = len(model.b1)

    with open(path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<4I", vocab_size, model.ctx, emb, hidden))

        # Biases are stored as one-row matrices so they use the same format
        tensors = [
            model.C,
            model.W1,
            [model.b1],
            model.W2,
            [model.b2],
        ]
        for rows in tensors:
            n_rows, n_cols = len(rows), len(rows[0])
            scales, packed = _quantize_rows(rows)
            f.write(struct.pack("<2I", n_rows, n_cols))
            f.write(struct.pack(f"<{n_rows}f", *scales))
            f.write(packed)


def load_model(path):
    """Rebuild a TinyLM from a file written by save_model."""
    with open(path, "rb") as f:
        if f.read(4) != MAGIC:
            raise ValueError(f"{path} is not a 4-bit TinyLM file")
        vocab_size, ctx, emb, hidden = struct.unpack("<4I", f.read(16))
        model = TinyLM(vocab_size=vocab_size, ctx=ctx, emb=emb, hidden=hidden)

        targets = [model.C, model.W1, [model.b1], model.W2, [model.b2]]
        for target in targets:
            n_rows, n_cols = struct.unpack("<2I", f.read(8))
            if n_rows != len(target) or n_cols != len(target[0]):
                raise ValueError("Shape mismatch while loading weights")
            scales = struct.unpack(f"<{n_rows}f", f.read(4 * n_rows))
            packed = f.read((n_rows * n_cols + 1) // 2)
            values = _dequantize_rows(scales, packed, n_rows, n_cols)
            for row, new_row in zip(target, values):
                for v, x in zip(row, new_row):
                    v.data = x

    return model

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


# ---- training and generation ----------------------------------------------

def train(model, ids, steps=2000, lr=0.05, batch_size=16, n_negatives=256):
    """Mini-batch SGD with sampled softmax.

    Each step averages the loss over `batch_size` windows. For each target, the
    softmax runs over the true token plus `n_negatives` random tokens instead of
    the whole vocabulary. This cuts the output-layer cost by roughly
    vocab_size / (n_negatives + 1).
    """
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


def generate(model, tok, prompt, n_tokens=60, temperature=0.8):
    ids = tok.encode(prompt) or [0]
    for _ in range(n_tokens):
        window = ids[-model.ctx:]
        window = [0] * (model.ctx - len(window)) + window
        logits = [l / temperature for l in fast_logits(model, window)]
        m = max(logits)
        weights = [math.exp(x - m) for x in logits]
        next_id = random.choices(range(len(weights)), weights=weights)[0]
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


def _init_worker(flat, cfg, vocab_size, n_negatives):
    """Initializer for persistent worker pool."""
    _WorkerState.model = TinyLM(
        vocab_size=cfg["vocab"],
        ctx=cfg["ctx"],
        emb=cfg["emb"],
        hidden=cfg["hidden"],
    )
    _WorkerState.cfg = cfg
    _WorkerState.vocab_size = vocab_size
    _WorkerState.n_negatives = n_negatives
    _set_flat_params(_WorkerState.model, flat)


def _worker_grads_v2(args):
    """Worker function that receives (flat, windows) and uses persistent model."""
    flat, windows = args
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


def train_parallel(model, ids, steps=2000, lr=0.05, batch_size=16, n_negatives=256, workers=None):
    """Data-parallel version of train. Uses persistent worker pool to avoid
    rebuilding the model each step. The batch is split across processes and
    the gradients are summed before one SGD update."""
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