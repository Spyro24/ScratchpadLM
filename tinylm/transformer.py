"""Transformer primitives: LayerNorm, attention, FFN, TransformerBlock."""
import math
import random
from typing import List, Optional

from tinylm.autograd import Value, matvec_dot

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

