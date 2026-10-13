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


def generate(model, tok, prompt, max_tokens=60, min_tokens=0, temperature=0.8, 
             top_k=0, top_p=0.0, repetition_penalty=1.0,
             stop_token_ids=None, stop_strings=None, stream_callback=None,
             **kwargs):
    """Generate text from prompt with autostop support.
    
    Args:
        model: Model instance
        tok: Tokenizer
        prompt: Input prompt string
        max_tokens: Maximum tokens to generate (default 60). Can also use deprecated `n_tokens`.
        min_tokens: Minimum tokens before autostop can trigger (default 0)
        temperature: Sampling temperature
        top_k: Top-k sampling parameter (0 = disabled)
        top_p: Top-p (nucleus) sampling parameter (0 = disabled)
        repetition_penalty: Repetition penalty (>1.0)
        stop_token_ids: Optional set/list of token IDs that trigger stop
        stop_strings: Optional list of strings that trigger stop when generated
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
    
    ids = tok.encode(prompt) or [0]
    model_fast_logits = getattr(model, 'fast_logits', None)
    
    # Convert stop_token_ids to set for O(1) lookup
    stop_tokens = set(stop_token_ids) if stop_token_ids else None
    
    # Pre-encode stop strings to token sequences
    stop_sequences = []
    if stop_strings:
        for s in stop_strings:
            seq = tok.encode(s)
            if seq:
                stop_sequences.append(seq)
    
    generated_text = ""
    start_time = time.time()
    
    for i in range(max_tokens):
        window = ids[-model.ctx:]
        window = [0] * (model.ctx - len(window)) + window
        if model_fast_logits is not None:
            logits = model_fast_logits(window)
        else:
            logits = fast_logits(model, window)
        next_id, should_stop = _sample_next_token(logits, temperature, top_k, top_p, 
                                                   repetition_penalty, ids, stop_tokens)
        ids.append(next_id)
        
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
            tps = (i + 1) / elapsed if elapsed > 0 else 0
            # Callback can return False to stop early
            if stream_callback(token_str, i + 1, max_tokens, tps) is False:
                should_stop = True
        
        # Check stop conditions
        if should_stop and i + 1 >= min_tokens:
            break
    
    return tok.decode(ids)


def generate_with_preview(model, tok, prompt, max_tokens=60, min_tokens=0, temperature=0.8, top_k=0, top_p=0.0, 
                          repetition_penalty=1.0, stop_token_ids=None, stop_strings=None) -> dict:
    """Generate text with live token preview and timing stats.
    
    Args:
        model: Model instance
        tok: Tokenizer
        prompt: Input prompt string
        max_tokens: Maximum tokens to generate (default 60)
        min_tokens: Minimum tokens before autostop can trigger (default 0)
        temperature: Sampling temperature
        top_k: Top-k sampling parameter (0 = disabled)
        top_p: Top-p (nucleus) sampling parameter (0 = disabled)
        repetition_penalty: Repetition penalty (>1.0)
        stop_token_ids: Optional set/list of token IDs that trigger stop
        stop_strings: Optional list of strings that trigger stop when generated

    Returns:
        Dict with 'text', 'tokens_generated', 'elapsed_sec', 'tokens_per_sec', 'tokens'
    """
    import time
    
    # Backward compatibility
    if 'n_tokens' in locals() or 'n_tokens' in globals():
        pass  # max_tokens is the primary parameter now
    
    ids = tok.encode(prompt) or [0]
    model_fast_logits = getattr(model, 'fast_logits', None)
    
    # Convert stop_token_ids to set for O(1) lookup
    stop_tokens = set(stop_token_ids) if stop_token_ids else None
    
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
    
    print(f"\n{'='*60}")
    print(f"Generating up to {max_tokens} tokens...")
    print(f"Prompt: {prompt[:80]}{'...' if len(prompt) > 80 else ''}")
    print(f"{'='*60}")
    
    for i in range(max_tokens):
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
        if (i + 1) % 10 == 0 or (i + 1) == max_tokens:
            elapsed = time.time() - start_time
            tps = (i + 1) / elapsed if elapsed > 0 else 0
            print(f"\n  [{i+1}/{max_tokens}] {tps:.1f} tok/s")
        
        # Check stop conditions
        if should_stop and i + 1 >= min_tokens:
            print(f"\n  [Stopped early at token {i+1}]")
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
