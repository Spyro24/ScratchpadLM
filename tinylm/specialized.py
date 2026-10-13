"""SpecializedLM: 1-layer transformer base (<=100K params) + presets."""
import random
from dataclasses import dataclass
from typing import Dict, List

from tinylm.autograd import Value, matvec_dot, softmax_cross_entropy
from tinylm.transformer import TransformerBlock

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
        from tinylm.serialization import count_params, enforce_param_limit
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
    
    def sampled_loss(self, context: List[int], target: int, n_negatives: int, vocab_size: int, negatives=None) -> Value:
        """Sampled softmax loss for training."""
        x = self._embed_context(context)
        for block in self.blocks:
            x = block.forward(x)
        
        if negatives is not None:
            candidates = [target] + negatives
        else:
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

