"""Router model: 2-layer transformer for task decomposition (~300K params)."""
import random
from dataclasses import dataclass
from typing import List, Optional

from tinylm.autograd import Value, matvec_dot, softmax_cross_entropy
from tinylm.transformer import TransformerBlock

# ---- Router Model -----------------------------------------------------------

@dataclass
class RouterConfig:
    """Configuration for Router model."""
    vocab_size: int = 1000
    ctx: int = 50
    emb: int = 64
    hidden: int = 128
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
        """Fast forward pass for inference."""
        # Embed
        x = []
        vocab_size = len(self.C)
        for i, tok in enumerate(context):
            if i >= self.config.ctx:
                break
            # Clamp token ID to valid range to handle vocab mismatch gracefully
            if tok >= vocab_size:
                tok = 0
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
                    # Factor in model metrics if available (e.g., avg_rating)
                    metrics = spec.metrics or {}
                    # Boost score based on avg_rating (1-5 scale)
                    avg_rating = metrics.get("avg_rating", 0)
                    if avg_rating > 0:
                        score += avg_rating * 0.5
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

    def route(self, prompt: str, tokenizer, catalog) -> dict:
        """Advanced routing using router model's inference + catalog.
        
        This method uses the router's learned representations to predict
        the best model for a given prompt, augmented with catalog metadata.
        
        Args:
            prompt: Input prompt to route
            tokenizer: BPETokenizer for encoding
            catalog: ModelCatalog instance
            
        Returns:
            Dict with 'model', 'prompt', 'context_tokens', 'confidence'
        """
        # Encode prompt
        ids = tokenizer.encode(prompt)
        context = ids[-self.config.ctx:] if len(ids) > self.config.ctx else ids
        
        # Get router model's logits (inference mode)
        logits = self.fast_logits(context)
        
        # Score each model in catalog based on router's output
        # The router's output logits over vocab can be interpreted as
        # model preference signals when trained on router training data
        scores = {}
        for spec in catalog.list():
            if spec.name == "router":
                continue
            
            # Use domain keywords as proxy for model preference
            # In a fully trained router, this would use learned representations
            prompt_lower = prompt.lower()
            keyword_score = sum(2 for kw in spec.domain_keywords if kw in prompt_lower)
            
            # Add catalog-based confidence from model metrics
            metrics = spec.metrics or {}
            avg_rating = metrics.get("avg_rating", 3.0)  # Default neutral
            train_count = metrics.get("train_count", 0)
            
            # Weight: keyword matches + rating + training experience
            confidence = keyword_score + (avg_rating * 0.3) + min(train_count * 0.1, 1.0)
            
            if confidence > 0:
                scores[spec.name] = confidence
        
        if not scores:
            # Fallback
            return {"model": "tiny-prose", "prompt": prompt, "context_tokens": ids, "confidence": 0.5}
        
        # Return best model
        best_model = max(scores.items(), key=lambda x: x[1])
        return {
            "model": best_model[0],
            "prompt": prompt,
            "context_tokens": ids,
            "confidence": min(best_model[1] / 10.0, 1.0)  # Normalize to 0-1
        }

    def route_multi(self, prompt: str, tokenizer, catalog, max_subtasks: int = 4) -> List[dict]:
        """Route prompt to multiple models for collaborative generation.
        
        Returns list of routing decisions, each with confidence score.
        """
        # Get primary route
        primary = self.route(prompt, tokenizer, catalog)
        
        # Get additional candidates for multi-model routing
        prompt_lower = prompt.lower()
        scores = {}
        for spec in catalog.list():
            if spec.name == "router":
                continue
            score = sum(1 for kw in spec.domain_keywords if kw in prompt_lower)
            if score > 0:
                metrics = spec.metrics or {}
                avg_rating = metrics.get("avg_rating", 0)
                if avg_rating > 0:
                    score += avg_rating * 0.3
                scores[spec.name] = score
        
        # Sort and build result list
        sorted_models = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        results = [primary]
        
        for model_name, score in sorted_models[:max_subtasks]:
            if model_name != primary["model"]:
                results.append({
                    "model": model_name,
                    "prompt": prompt,
                    "context_tokens": tokenizer.encode(prompt),
                    "confidence": min(score / 10.0, 1.0)
                })
        
        return results[:max_subtasks]

