"""RWKV-style Linear Attention model for fast training (stdlib only)."""

import random
import math
from typing import List, Optional

from tinylm.autograd import Value, matvec_dot, softmax_cross_entropy


class LinearAttention:
    """RWKV-style linear attention block.
    
    Uses the WKV (Weighted Key-Value) formulation for linear attention:
    w = exp(-exp(log_w))  # decay
    k = exp(k)            # key
    v = v                 # value
    
    wkv = (sum w^i * k_i * v_i) / (sum w^i * k_i)
    """
    
    def __init__(self, hidden: int, ctx: int, seed: int = 0):
        rng = random.Random(seed)
        
        def matrix(rows, cols, scale=0.1):
            return [[Value(rng.gauss(0.0, scale)) for _ in range(cols)] for _ in range(rows)]
        
        def vector(size, scale=0.1):
            return [Value(rng.gauss(0.0, scale)) for _ in range(size)]
        
        # Time decay (log_w) - learned per channel
        self.log_w = vector(hidden, scale=0.01)
        
        # Time decay for key (learned per channel)
        self.log_k = vector(hidden, scale=0.01)
        
        # Output projection
        self.W_out = matrix(hidden, hidden)
        self.b_out = vector(hidden)
        
        self.params = (
            self.log_w + self.log_k
            + [p for row in self.W_out for p in row] + self.b_out
        )
    
    def forward(self, x: List[Value], state: Optional[List[Value]] = None) -> tuple:
        """Forward pass with recurrent state (single time step).

        Args:
            x: Input vector (hidden)
            state: Previous state [wkv_num, wkv_den, wkv_max] or None

        Returns:
            (output, new_state)
        """
        hidden = len(x)

        # Time decay and key as graph ops so log_w/log_k receive gradients.
        # w = -exp(-exp(log_w)), k = exp(log_k) * x  (max stays constant).
        w = []
        for v in self.log_w:
            e_inner = v.exp()           # exp(log_w)
            e_outer = (-e_inner).exp()  # exp(-exp(log_w))
            w.append(-e_outer)          # -exp(-exp(log_w))
        k = [(v.exp()) * xi for v, xi in zip(self.log_k, x)]
        v = x

        # Initialize or update state
        if state is None:
            wkv_num = [Value(0.0) for _ in range(hidden)]
            wkv_den = [Value(0.0) for _ in range(hidden)]
            wkv_max = [Value(-1e30) for _ in range(hidden)]
        else:
            wkv_num, wkv_den, wkv_max = state[:hidden], state[hidden:2*hidden], state[2*hidden:]

        # Update WKV state (RWKV formulation)
        for i in range(hidden):
            # New max for numerical stability (constant: no gradient through max)
            new_max = Value(max(wkv_max[i].data, k[i].data))

            # Exponential scaling (constants)
            e1 = Value(math.exp(wkv_max[i].data - new_max.data))
            e2 = Value(math.exp(k[i].data - new_max.data))

            # Update numerator and denominator (graph ops: grads flow to
            # num/den/w->log_w and v,k->x,log_k)
            ew = w[i].exp()
            wkv_num[i] = wkv_num[i] * ew * e1 + v[i] * k[i] * e2
            wkv_den[i] = wkv_den[i] * ew * e1 + k[i] * e2
            wkv_max[i] = new_max

        # Compute output
        wkv = [wkv_num[i] / (wkv_den[i] + Value(1e-8)) for i in range(hidden)]

        # Output projection
        out = matvec_dot(self.W_out, wkv, self.b_out)

        new_state = wkv_num + wkv_den + wkv_max
        return out, new_state

    def forward_arrays(self, x: List[float], state: Optional[List[float]] = None) -> tuple:
        """Fast inference on plain floats (no autograd)."""
        import math
        hidden = len(x)
        
        w = [-math.exp(-math.exp(self.log_w[i].data)) for i in range(hidden)]
        k = [math.exp(self.log_k[i].data) * x[i] for i in range(hidden)]
        v = x
        
        if state is None:
            wkv_num = [0.0] * hidden
            wkv_den = [0.0] * hidden
            wkv_max = [-1e30] * hidden
        else:
            wkv_num, wkv_den, wkv_max = state[:hidden], state[hidden:2*hidden], state[2*hidden:]
        
        for i in range(hidden):
            new_max = max(wkv_max[i], k[i])
            e1 = math.exp(wkv_max[i] - new_max)
            e2 = math.exp(k[i] - new_max)
            wkv_num[i] = wkv_num[i] * math.exp(w[i]) * e1 + v[i] * k[i] * e2
            wkv_den[i] = wkv_den[i] * math.exp(w[i]) * e1 + k[i] * e2
            wkv_max[i] = new_max
        
        wkv = [wkv_num[i] / (wkv_den[i] + 1e-8) for i in range(hidden)]
        
        # Output projection
        out = [0.0] * hidden
        for i in range(hidden):
            s = self.b_out[i].data
            for j, w in enumerate(self.W_out[i]):
                s += w.data * wkv[j]
            out[i] = s
        
        new_state = wkv_num + wkv_den + wkv_max
        return out, new_state


class RWKVBlock:
    """RWKV-style block: Time-mixing + Channel-mixing."""
    
    def __init__(self, hidden: int, ctx: int, seed: int = 0):
        rng = random.Random(seed)
        
        def matrix(rows, cols, scale=0.1):
            return [[Value(rng.gauss(0.0, scale)) for _ in range(cols)] for _ in range(rows)]
        
        def vector(size, scale=0.1):
            return [Value(rng.gauss(0.0, scale)) for _ in range(size)]
        
        # Time-mixing (attention)
        self.attn = LinearAttention(hidden, ctx, seed)
        
        # Layer norm 1
        self.ln1_w = vector(hidden, scale=1.0)
        self.ln1_b = vector(hidden, scale=0.0)
        
        # Channel-mixing (FFN)
        self.ffn_W1 = matrix(hidden * 4, hidden)
        self.ffn_b1 = vector(hidden * 4)
        self.ffn_W2 = matrix(hidden, hidden * 4)
        self.ffn_b2 = vector(hidden)
        
        # Layer norm 2
        self.ln2_w = vector(hidden, scale=1.0)
        self.ln2_b = vector(hidden, scale=0.0)
        
        # Time-mixing params
        self.attn_lora = vector(hidden, scale=0.01)
        
        self.params = (
            self.attn.params
            + self.ln1_w + self.ln1_b
            + [p for row in self.ffn_W1 for p in row] + self.ffn_b1
            + [p for row in self.ffn_W2 for p in row] + self.ffn_b2
            + self.ln2_w + self.ln2_b
            + self.attn_lora
        )
    
    def layer_norm(self, x: List[Value], w: List[Value], b: List[Value]) -> List[Value]:
        """Layer normalization."""
        mean = sum(x) / len(x)
        var = sum((xi - mean) * (xi - mean) for xi in x) / len(x)
        return [(xi - mean) * w[i] / (var + Value(1e-5)) + b[i] for i, xi in enumerate(x)]
    
    def forward(self, x: List[Value], state: Optional[List[Value]] = None) -> tuple:
        """Forward pass: Time-mixing + Channel-mixing."""
        hidden = len(x)
        
        # Split state
        if state is None:
            attn_state = None
        else:
            attn_state = state[:3 * hidden]
        
        # Time-mixing (Linear Attention)
        ln1_x = self.layer_norm(x, self.ln1_w, self.ln1_b)
        attn_out, new_attn_state = self.attn.forward(ln1_x, attn_state)
        
        # Residual with LoRA-style mixing
        x = [x[i] + attn_out[i] for i in range(hidden)]
        
        # Channel-mixing (FFN)
        ln2_x = self.layer_norm(x, self.ln2_w, self.ln2_b)
        h = matvec_dot(self.ffn_W1, ln2_x, self.ffn_b1)
        # GELU approximation
        h = [hi * (Value(0.5) * (Value(1.0) + (hi * Value(0.7978845608028654)).tanh())) for hi in h]
        ffn_out = matvec_dot(self.ffn_W2, h, self.ffn_b2)
        
        # Residual
        x = [x[i] + ffn_out[i] for i in range(hidden)]
        
        return x, new_attn_state


class RWKVModel:
    """RWKV-style model: Embedding -> RWKV blocks -> Output.
    
    Architecture:
    - Token embeddings
    - Multiple RWKV blocks (time-mixing + channel-mixing)
    - Output projection to vocab
    """
    
    def __init__(self, vocab_size: int, ctx: int = 50, emb: int = 128, hidden: int = 256, 
                 n_layers: int = 4, seed: int = 0):
        rng = random.Random(seed)
        self.ctx = ctx
        self.vocab_size = vocab_size
        self.hidden = hidden
        self.n_layers = n_layers
        
        def matrix(rows, cols, scale=0.1):
            return [[Value(rng.gauss(0.0, scale)) for _ in range(cols)] for _ in range(rows)]
        
        def vector(size, scale=0.1):
            return [Value(rng.gauss(0.0, scale)) for _ in range(size)]
        
        # Embeddings
        self.C = matrix(vocab_size, emb)
        
        # Input projection if emb != hidden
        if emb != hidden:
            self.in_proj = matrix(hidden, emb)
            self.in_bias = vector(hidden)
        else:
            self.in_proj = None
            self.in_bias = None
        
        # RWKV blocks
        self.blocks = [RWKVBlock(hidden, ctx, seed + i) for i in range(n_layers)]
        
        # Output projection
        self.out_proj = matrix(vocab_size, hidden)
        self.out_bias = vector(vocab_size)
        
        # Collect all params
        self.params = (
            [p for row in self.C for p in row]
            + ([p for row in self.in_proj for p in row] if self.in_proj else [])
            + (self.in_bias if self.in_bias else [])
        )
        for block in self.blocks:
            self.params += block.params
        self.params += [p for row in self.out_proj for p in row] + self.out_bias
    
    def get_config(self) -> dict:
        return {
            "arch": "RWKVModel",
            "vocab_size": self.vocab_size,
            "ctx": self.ctx,
            "emb": len(self.C[0]) if self.C else 0,
            "hidden": self.hidden,
            "n_layers": self.n_layers,
        }

    def get_tensors(self) -> dict:
        """All live tensors (bias vectors wrapped as 1-row matrices).

        Includes every RWKV block so save/load round-trips all weights.
        """
        tensors = {"C": self.C}
        if self.in_proj is not None:
            tensors["in_proj"] = self.in_proj
            tensors["in_bias"] = [self.in_bias]
        for i, block in enumerate(self.blocks):
            p = f"b{i}_"
            tensors[p + "ln1_w"] = [block.ln1_w]
            tensors[p + "ln1_b"] = [block.ln1_b]
            tensors[p + "attn_log_w"] = [block.attn.log_w]
            tensors[p + "attn_log_k"] = [block.attn.log_k]
            tensors[p + "attn_W_out"] = block.attn.W_out
            tensors[p + "attn_b_out"] = [block.attn.b_out]
            tensors[p + "ffn_W1"] = block.ffn_W1
            tensors[p + "ffn_b1"] = [block.ffn_b1]
            tensors[p + "ffn_W2"] = block.ffn_W2
            tensors[p + "ffn_b2"] = [block.ffn_b2]
            tensors[p + "ln2_w"] = [block.ln2_w]
            tensors[p + "ln2_b"] = [block.ln2_b]
            tensors[p + "attn_lora"] = [block.attn_lora]
        tensors["out_proj"] = self.out_proj
        tensors["out_bias"] = [self.out_bias]
        return tensors

    def fast_logits(self, context: List[int]) -> List[float]:
        """Fast inference on plain floats (no autograd).

        Mirrors logits(): recurrent over context tokens, same formulas
        (including the no-sqrt layer norm) so inference matches training.
        """
        import math
        if not context:
            raise ValueError("fast_logits() needs a non-empty context")

        states = [None] * self.n_layers
        x = None
        for tok in context:
            if tok >= self.vocab_size:
                tok = 0
            emb = [v.data for v in self.C[tok]]
            if self.in_proj is not None:
                proj = [0.0] * self.hidden
                for i, row in enumerate(self.in_proj):
                    s = self.in_bias[i].data
                    for j, w in enumerate(row):
                        s += w.data * emb[j]
                    proj[i] = s
                x = proj
            else:
                x = emb

            new_states = []
            for li, block in enumerate(self.blocks):
                # LayerNorm 1 (same no-sqrt formula as training)
                ln1_x = self._layer_norm_arrays(x, block.ln1_w, block.ln1_b)
                attn_out, new_attn = block.attn.forward_arrays(
                    ln1_x, states[li])
                x = [x[i] + attn_out[i] for i in range(self.hidden)]
                # Channel-mixing
                ln2_x = self._layer_norm_arrays(x, block.ln2_w, block.ln2_b)
                h = [0.0] * (self.hidden * 4)
                for i, row in enumerate(block.ffn_W1):
                    s = block.ffn_b1[i].data
                    for j, w in enumerate(row):
                        s += w.data * ln2_x[j]
                    h[i] = s
                # GELU (same tanh form as training)
                h = [hi * 0.5 * (1.0 + math.tanh(hi * 0.7978845608028654)) for hi in h]
                ffn_out = [0.0] * self.hidden
                for i, row in enumerate(block.ffn_W2):
                    s = block.ffn_b2[i].data
                    for j, w in enumerate(row):
                        s += w.data * h[j]
                    ffn_out[i] = s
                x = [x[i] + ffn_out[i] for i in range(self.hidden)]
                new_states.append(new_attn)
            states = new_states

        # Output projection
        logits = [0.0] * self.vocab_size
        for i, row in enumerate(self.out_proj):
            s = self.out_bias[i].data
            for j, w in enumerate(row):
                s += w.data * x[j]
            logits[i] = s
        return logits

    def gpu_fast_logits(self, context: List[int]) -> List[float]:
        """GPU logits via accel_rwkv (raises when unavailable/unimplemented).

        ``tinylm/generation.py`` calls this first and falls back to the CPU
        ``fast_logits`` on any exception, so CPU fallback always stays.
        """
        try:
            import accel_rwkv
        except Exception as exc:
            raise RuntimeError(f"accel_rwkv unavailable: {exc}")
        flat = [p.data for p in self.params]
        return accel_rwkv.rwkv_generate_step(
            flat, list(context), self.vocab_size,
            len(self.C[0]), self.hidden, self.n_layers, self.ctx,
        )

    def _layer_norm_arrays(self, x, w, b):
        # Same formula as RWKVBlock.layer_norm (no sqrt) so inference
        # matches training exactly.
        mean = sum(x) / len(x)
        var = sum((xi - mean) ** 2 for xi in x) / len(x)
        return [(xi - mean) * w[i].data / (var + 1e-5) + b[i].data for i, xi in enumerate(x)]
    
    def _embed_token(self, tok: int) -> List[Value]:
        """Embed a single token, projected to hidden dim."""
        emb = list(self.C[tok])
        if self.in_proj is not None:
            return matvec_dot(self.in_proj, emb, self.in_bias)
        return emb

    def _forward_hidden(self, context: List[int], state: Optional[List[List[Value]]] = None) -> tuple:
        """Recurrent pass returning the final hidden state and new WKV states."""
        states = list(state) if state is not None else [None] * self.n_layers
        x = None
        for tok in context:
            x = self._embed_token(tok)
            new_states = []
            for i, block in enumerate(self.blocks):
                x, new_state = block.forward(x, states[i])
                new_states.append(new_state)
            states = new_states
        if x is None:
            raise ValueError("logits() needs a non-empty context")
        return x, states

    def logits(self, context: List[int], state: Optional[List[List[Value]]] = None) -> tuple:
        """Forward pass returning logits and new states.

        Recurrent over context tokens: each token is embedded, run through
        every block (carrying that block's WKV state), and only the final
        token's output is projected to logits. Returned states continue
        generation without reprocessing the prompt.
        """
        x, states = self._forward_hidden(context, state)
        logits = matvec_dot(self.out_proj, x, self.out_bias)
        return logits, states
    
    def loss(self, context: List[int], target: int, state: Optional[List[List[Value]]] = None) -> tuple:
        logits, new_state = self.logits(context, state)
        return softmax_cross_entropy(logits, target), new_state
    
    def sampled_loss(self, context: List[int], target: int, n_negatives: int,
                     vocab_size: int, negatives: Optional[List[int]] = None,
                     state: Optional[List[List[Value]]] = None) -> tuple:
        # Project the final hidden state through the candidate rows only
        # (mirrors TinyLM.sampled_loss). Projecting the full logits vector
        # instead would silently truncate at min(hidden, vocab).
        x, new_state = self._forward_hidden(context, state)

        if negatives is not None:
            candidates = [target] + negatives
        else:
            candidates = [target] + [random.randrange(vocab_size) for _ in range(n_negatives)]

        W2_sub = [self.out_proj[c] for c in candidates]
        b2_sub = [self.out_bias[c] for c in candidates]
        logits_sub = matvec_dot(W2_sub, x, b2_sub)
        return softmax_cross_entropy(logits_sub, 0), new_state