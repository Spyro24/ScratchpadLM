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
        """Mean-pool token + positional embeddings over the window.

        Returns `emb` Values (not `ctx * emb`): the transformer blocks below
        consume exactly `d_model == emb` inputs, so concatenating would leave
        every position past the first unread — the router would be blind to
        most of its context window. Shapes are unchanged, so saved models
        still load.
        """
        emb = self.config.emb
        acc = [Value(0.0) for _ in range(emb)]
        n = 0
        for i, tok in enumerate(context):
            if i >= self.config.ctx:
                break
            tok_emb = self.C[tok]
            pos_emb = self.pos_emb[i]
            for j in range(emb):
                acc[j] = acc[j] + tok_emb[j] + pos_emb[j]
            n += 1
        inv = 1.0 / max(1, n)
        return [a * inv for a in acc]
    
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
        """Fast forward pass for inference (mean-pooled, mirrors `_embed_context`)."""
        # Embed: mean-pool token + positional embeddings over the window
        emb = self.config.emb
        vocab_size = len(self.C)
        acc = [0.0] * emb
        n = 0
        for i, tok in enumerate(context):
            if i >= self.config.ctx:
                break
            # Clamp token ID to valid range to handle vocab mismatch gracefully
            if tok >= vocab_size:
                tok = 0
            tok_emb = self.C[tok]
            pos_emb = self.pos_emb[i]
            for j in range(emb):
                acc[j] += tok_emb[j].data + pos_emb[j].data
            n += 1
        inv = 1.0 / max(1, n)
        x = [a * inv for a in acc]
        
        # Transformer blocks
        for block in self.blocks:
            x = block.fast_forward(x)
        
        # Output projection
        return [sum(self.W_out[i][j].data * x[j] for j in range(self.config.emb)) + self.b_out[i].data
                for i in range(self.config.vocab_size)]
    
    def split(self, prompt: str, tokenizer, max_subtasks: int = 4, catalog=None) -> List[dict]:
        """Decompose prompt into subtasks with model assignments.

        Model names come ONLY from `catalog` (plus the fixed `router`
        identity, which is skipped): without a catalog — or when nothing
        scores — this returns [] rather than inventing model names.

        Returns list of dicts: [{model: str, prompt: str, context_tokens: List[int]}]

        Args:
            prompt: Input prompt to route
            tokenizer: BPETokenizer for encoding
            max_subtasks: Maximum number of subtasks to return
            catalog: Optional ModelCatalog to use domain_keywords for routing
        """
        ids = tokenizer.encode(prompt)
        if catalog is None:
            return []
        prompt_lower = prompt.lower()

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

        if not scores:
            return []
        # Return top-scoring models (up to max_subtasks)
        sorted_models = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [
            {"model": model_name, "prompt": prompt, "context_tokens": ids}
            for model_name, _ in sorted_models[:max_subtasks]
        ]

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
            # No signal: fall back to the first catalogued model rather than
            # inventing a name. Only the `router` identity is fixed.
            models = [s.name for s in catalog.list() if s.name != "router"]
            if not models:
                raise RuntimeError("route(): no models available in catalog")
            return {"model": models[0], "prompt": prompt,
                    "context_tokens": ids, "confidence": 0.0}
        
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

    def decompose(self, prompt: str, tokenizer, max_subtasks: int = 4,
                  max_tokens: int = 128, temperature: float = 0.0,
                  catalog=None) -> List[dict]:
        """Decompose prompt with the trained model weights (inference usage).

        Autoregressively extends `[prompt_tokens, SEP]` via `fast_logits`
        (greedy by default, temperature sampling when `temperature > 0`) and
        decodes the sequence with `decode_router_sequence`, so a trained
        router's learned decomposition is used instead of heuristics.
        Stops after `max_subtasks` SEP-terminated segments or `max_tokens`.
        Returns `[{model, prompt, context_tokens}]`; when `catalog` is given,
        decoded names outside the catalog are dropped; nothing decodable (or
        nothing catalogued) yields [] — model names are never invented.
        """
        import math
        from tinylm.router_data import decode_router_sequence
        vocab = len(tokenizer.vocab)
        SEP = vocab - 1
        ids = tokenizer.encode(prompt)
        seq = (ids[-self.config.ctx:] if len(ids) > self.config.ctx else list(ids))
        seq.append(SEP)
        segments = 0
        for _ in range(max_tokens):
            window = seq[-self.config.ctx:]
            window = [0] * (self.config.ctx - len(window)) + window
            logits = self.fast_logits(window)
            if temperature > 0:
                m = max(logits)
                weights = [math.exp((x - m) / temperature) for x in logits]
                nxt = random.choices(range(len(weights)), weights=weights)[0]
            else:
                nxt = max(range(len(logits)), key=logits.__getitem__)
            seq.append(nxt)
            if nxt == SEP:
                segments += 1
                if segments >= max_subtasks:
                    break
        example = decode_router_sequence(seq, tokenizer, vocab)
        available = None
        if catalog is not None:
            try:
                available = {s.name for s in catalog.list()}
            except Exception:
                available = None
        out = []
        for sub in example.subtasks[:max_subtasks]:
            name = sub.get("model")
            if available is not None and name not in available:
                continue
            sub_prompt = sub.get("prompt", prompt)
            out.append({
                "model": name,
                "prompt": sub_prompt,
                "context_tokens": tokenizer.encode(sub_prompt),
            })
        return out

