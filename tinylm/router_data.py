"""Router training data format, encoding, pretraining helpers."""
import itertools
import re
from dataclasses import dataclass
from typing import Dict, List, Optional

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
    from tinylm.training import train
    train(router, ids, steps=steps, lr=lr, batch_size=batch_size,
          n_negatives=n_negatives, progress_cb=progress_cb, **train_kwargs)
    
    return router


def generate_router_training_data(corpus: str, tokenizer, vocab_size: int, ctx: int,
                                   domain_keywords: Optional[dict] = None,
                                   max_subtasks: int = 3) -> List[RouterTrainingExample]:
    """Generate router training examples from a corpus using heuristic domain detection.

    Args:
        corpus: Text corpus to generate examples from
        tokenizer: BPETokenizer for encoding
        vocab_size: Vocabulary size (for special token allocation)
        ctx: Context window size
        domain_keywords: Optional dict mapping model names to keyword lists
                        e.g., {"tiny-code": ["code", "function", "python"],
                               "tiny-reason": ["math", "calculate", "logic"]}
        max_subtasks: Max subtasks per composite example (1 = single-subtask
                      examples only, as before)

    Returns:
        List of RouterTrainingExample objects: one single-subtask example per
        sentence, plus composite multi-subtask examples built from consecutive
        sentences spanning at least two distinct domains (teaches multi-step
        decomposition).
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
    labeled = []
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
        labeled.append((sent, best_model))
        examples.append(RouterTrainingExample(
            prompt=sent,
            subtasks=[{"model": best_model, "prompt": sent}]
        ))

    # Composite multi-subtask examples from consecutive sentences spanning
    # at least two distinct domains (teaches multi-step decomposition).
    for i in range(len(labeled)):
        group = labeled[i:i + max_subtasks]
        if len(group) < 2 or len({m for _, m in group}) < 2:
            continue
        examples.append(RouterTrainingExample(
            prompt=" ".join(s for s, _ in group),
            subtasks=[{"model": m, "prompt": s} for s, m in group],
        ))

    return examples

