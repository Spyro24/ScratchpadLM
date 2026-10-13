"""Text generation: fast inference, sampling, autostop (stdlib only)."""
import math
import random
import time
from typing import List, Optional, Tuple

def fast_logits(model, context):
    """Forward pass on plain floats, no graph. Used for generation.
    
    Optimized: uses local variables, avoids repeated attribute access.
    """
    # Embed context
    C = model.C
    vocab_size = len(C)
    x = []
    for tok in context:
        # Clamp token ID to valid range to handle vocab mismatch gracefully
        if tok >= vocab_size:
            tok = 0  # Use padding token as fallback
        x.extend(v.data for v in C[tok])
    
    # Fused: h = tanh(W1 @ x + b1)
    W1 = model.W1
    b1 = model.b1
    h = [0.0] * len(W1)
    for i, (row, b) in enumerate(zip(W1, b1)):
        s = b.data
        for w, xi in zip(row, x):
            s += w.data * xi
        h[i] = math.tanh(s)
    
    # Fused: logits = W2 @ h + b2
    W2 = model.W2
    b2 = model.b2
    logits = [0.0] * len(W2)
    for i, (row, b) in enumerate(zip(W2, b2)):
        s = b.data
        for w, hi in zip(row, h):
            s += w.data * hi
        logits[i] = s
    return logits


def _sample_next_token(logits: list, temperature: float, top_k: int, top_p: float, 
                       repetition_penalty: float, generated_ids: list,
                       stop_token_ids=None) -> tuple:
    """Sample next token with temperature, top-k, top-p, and repetition penalty.
    
    Returns (next_token_id, should_stop).
    
    Optimized: caches softmax, uses efficient sorting, avoids redundant computations.
    """
    # Apply repetition penalty
    if repetition_penalty != 1.0:
        seen = set(generated_ids)
        for idx in seen:
            logits[idx] /= repetition_penalty

    # Apply temperature
    if temperature != 1.0:
        inv_temp = 1.0 / temperature
        for i in range(len(logits)):
            logits[i] *= inv_temp

    # Apply top-k
    if top_k > 0 and top_k < len(logits):
        # Use nsmallest/nlargest via sorting (stdlib only)
        # Get indices of top-k values
        indexed = list(enumerate(logits))
        indexed.sort(key=lambda x: x[1], reverse=True)
        top_k_indices = {idx for idx, _ in indexed[:top_k]}
        neg_inf = -float('inf')
        for i in range(len(logits)):
            if i not in top_k_indices:
                logits[i] = neg_inf

    # Apply top-p (nucleus sampling)
    if top_p > 0.0 and top_p < 1.0:
        # Single softmax pass for top-p
        m = max(logits)
        exps = [math.exp(x - m) for x in logits]
        s = sum(exps)
        # Create indexed list with probabilities
        indexed_probs = [(i, e / s) for i, e in enumerate(exps)]
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
        allowed = {idx for idx, _ in indexed_probs[:cutoff_idx]}
        neg_inf = -float('inf')
        for i in range(len(logits)):
            if i not in allowed:
                logits[i] = neg_inf

    # Final softmax and sample
    m = max(logits)
    weights = [math.exp(x - m) for x in logits]
    next_id = random.choices(range(len(weights)), weights=weights)[0]
    
    # Check for stop token
    should_stop = False
    if stop_token_ids is not None and next_id in stop_token_ids:
        should_stop = True
    
    return next_id, should_stop


def normalize_cap(max_tokens):
    """Translate UI/CLI token caps: None stays unbounded; 0 (or negative)
    also means unbounded — structural markers end generation instead.
    Raises ValueError on non-numeric input.
    """
    if max_tokens is None:
        return None
    n = int(max_tokens)
    return None if n <= 0 else n


def _resolve_stop_tokens(model, stop_token_ids, stop_markers):
    """Merge stop-token IDs + structural marker names into a set (or None).

    Marker names (`EOL`/`EOS`/`EOT`) resolve to single token IDs against the
    model's vocab (`len(model.C)`); out-of-range markers are skipped.
    """
    stop_tokens = set(stop_token_ids) if stop_token_ids else set()
    if stop_markers:
        from tinylm.data_clean import marker_id
        vocab = len(model.C)
        for name in stop_markers:
            mid = marker_id(name, vocab)
            if mid >= 0:
                stop_tokens.add(mid)
    return stop_tokens or None


def generate(model, tok, prompt, max_tokens=None, min_tokens=0, temperature=0.8, 
             top_k=0, top_p=0.0, repetition_penalty=1.0,
             stop_token_ids=None, stop_strings=None, stop_markers=("EOS",),
             stream_callback=None,
             **kwargs):
    """Generate text from prompt with autostop support.
    
    Args:
        model: Model instance
        tok: Tokenizer
        prompt: Input prompt string
        max_tokens: Maximum tokens to generate (None = unbounded: generation
            ends on stop markers/strings/ids or `stream_callback`).
            Can also use deprecated `n_tokens`.
        min_tokens: Minimum tokens before autostop can trigger (default 0)
        temperature: Sampling temperature
        top_k: Top-k sampling parameter (0 = disabled)
        top_p: Top-p (nucleus) sampling parameter (0 = disabled)
        repetition_penalty: Repetition penalty (>1.0)
        stop_token_ids: Optional set/list of token IDs that trigger stop
        stop_strings: Optional list of strings that trigger stop when generated
        stop_markers: Marker names (`EOL`/`EOS`/`EOT`) that trigger stop,
            resolved to token IDs against the model's vocab.
            Default `("EOS",)` ends generation after a finished section.
            Pass None/() to disable.
        stream_callback: Optional callback(token_str, token_idx, total_generated, tokens_per_sec)
                         Return False to stop generation early.
    
    Returns:
        Generated text string (includes prompt if you want just completion, 
        use tok.decode on returned ids[len(prompt_ids):] for completion only)
    """
    # Backward compatibility: accept n_tokens as alias for max_tokens
    if 'n_tokens' in kwargs:
        max_tokens = kwargs.pop('n_tokens')
    if kwargs:
        raise TypeError(f"generate() got unexpected keyword arguments: {list(kwargs.keys())}")
    max_tokens = normalize_cap(max_tokens)
    
    ids = tok.encode(prompt) or [0]
    model_fast_logits = getattr(model, 'fast_logits', None)
    
    # Merge stop-token IDs + structural marker names for O(1) lookup
    stop_tokens = _resolve_stop_tokens(model, stop_token_ids, stop_markers)
    
    # Pre-encode stop strings to token sequences
    stop_sequences = []
    if stop_strings:
        for s in stop_strings:
            seq = tok.encode(s)
            if seq:
                stop_sequences.append(seq)
    
    generated_text = ""
    start_time = time.time()
    count = 0
    
    while max_tokens is None or count < max_tokens:
        window = ids[-model.ctx:]
        window = [0] * (model.ctx - len(window)) + window
        if model_fast_logits is not None:
            logits = model_fast_logits(window)
        else:
            logits = fast_logits(model, window)
        next_id, should_stop = _sample_next_token(logits, temperature, top_k, top_p, 
                                                   repetition_penalty, ids, stop_tokens)
        ids.append(next_id)
        count += 1
        
        # Check stop strings (must check after adding the token)
        if stop_sequences:
            for seq in stop_sequences:
                if len(ids) >= len(seq) and ids[-len(seq):] == seq:
                    should_stop = True
                    break
        
        # Decode and yield token for streaming
        if stream_callback is not None:
            token_str = tok.decode([next_id])
            generated_text += token_str
            elapsed = time.time() - start_time
            tps = count / elapsed if elapsed > 0 else 0
            # Callback can return False to stop early
            if stream_callback(token_str, count, max_tokens, tps) is False:
                should_stop = True
        
        # Check stop conditions
        if should_stop and count >= min_tokens:
            break
    
    return tok.decode(ids)


def generate_with_preview(model, tok, prompt, max_tokens=None, min_tokens=0, temperature=0.8, top_k=0, top_p=0.0, 
                          repetition_penalty=1.0, stop_token_ids=None, stop_strings=None,
                          stop_markers=("EOS",)) -> dict:
    """Generate text with live token preview and timing stats.
    
    Args:
        model: Model instance
        tok: Tokenizer
        prompt: Input prompt string
        max_tokens: Maximum tokens to generate (None = unbounded: ends on
            stop markers/strings/ids)
        min_tokens: Minimum tokens before autostop can trigger (default 0)
        temperature: Sampling temperature
        top_k: Top-k sampling parameter (0 = disabled)
        top_p: Top-p (nucleus) sampling parameter (0 = disabled)
        repetition_penalty: Repetition penalty (>1.0)
        stop_token_ids: Optional set/list of token IDs that trigger stop
        stop_strings: Optional list of strings that trigger stop when generated
        stop_markers: Marker names (`EOL`/`EOS`/`EOT`) that trigger stop.
            Default `("EOS",)` ends after a finished section.

    Returns:
        Dict with 'text', 'tokens_generated', 'elapsed_sec', 'tokens_per_sec', 'tokens'
    """
    import time
    
    # Backward compatibility
    if 'n_tokens' in locals() or 'n_tokens' in globals():
        pass  # max_tokens is the primary parameter now
    max_tokens = normalize_cap(max_tokens)
    
    ids = tok.encode(prompt) or [0]
    model_fast_logits = getattr(model, 'fast_logits', None)
    
    # Merge stop-token IDs + structural marker names for O(1) lookup
    stop_tokens = _resolve_stop_tokens(model, stop_token_ids, stop_markers)
    
    # Pre-encode stop strings to token sequences
    stop_sequences = []
    if stop_strings:
        for s in stop_strings:
            seq = tok.encode(s)
            if seq:
                stop_sequences.append(seq)
    
    generated_tokens = []
    generated_text = ""
    start_time = time.time()
    count = 0
    
    print(f"\n{'='*60}")
    cap = f"up to {max_tokens} tokens" if max_tokens is not None else "until a stop marker"
    print(f"Generating {cap}...")
    print(f"Prompt: {prompt[:80]}{'...' if len(prompt) > 80 else ''}")
    print(f"{'='*60}")
    
    while max_tokens is None or count < max_tokens:
        window = ids[-model.ctx:]
        window = [0] * (model.ctx - len(window)) + window
        if model_fast_logits is not None:
            logits = model_fast_logits(window)
        else:
            logits = fast_logits(model, window)
        next_id, should_stop = _sample_next_token(logits, temperature, top_k, top_p, 
                                                   repetition_penalty, ids, stop_tokens)
        ids.append(next_id)
        generated_tokens.append(next_id)
        count += 1
        
        # Check stop strings
        if stop_sequences:
            for seq in stop_sequences:
                if len(ids) >= len(seq) and ids[-len(seq):] == seq:
                    should_stop = True
                    break
        
        # Decode and print token
        token_str = tok.decode([next_id])
        generated_text += token_str
        print(token_str, end="", flush=True)
        
        # Progress report every 10 tokens
        if count % 10 == 0 or (max_tokens is not None and count == max_tokens):
            elapsed = time.time() - start_time
            tps = count / elapsed if elapsed > 0 else 0
            total = max_tokens if max_tokens is not None else "…"
            print(f"\n  [{count}/{total}] {tps:.1f} tok/s")
        
        # Check stop conditions
        if should_stop and count >= min_tokens:
            print(f"\n  [Stopped early at token {count}]")
            break
    
    total_elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Done: {len(generated_tokens)} tokens in {total_elapsed:.2f}s ({len(generated_tokens)/total_elapsed:.1f} tok/s)")
    
    return {
        "text": generated_text,
        "tokens_generated": len(generated_tokens),
        "elapsed_sec": total_elapsed,
        "tokens_per_sec": len(generated_tokens) / total_elapsed if total_elapsed > 0 else 0,
        "tokens": generated_tokens
    }
