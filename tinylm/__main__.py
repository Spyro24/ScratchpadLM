"""Train-from-scratch demo (run with ``python3 -m tinylm``)."""
import random

from tokenizer import BPETokenizer

from tinylm.generation import generate
from tinylm.models_tiny import TinyLM
from tinylm.serialization import load_model, save_model, sync_vocab
from tinylm.training import train


def main() -> None:
    random.seed(0)

    corpus = ("the quick brown fox jumps over the lazy dog. " * 100
              + "a small model learns patterns from text. " * 100)

    tok = BPETokenizer()
    tok.train(corpus, vocab_size=30000)
    ids = tok.encode(corpus)
    try:
        model = load_model("./tinyLM")
        sync_vocab(model, len(tok.vocab))
        if len(ids) <= model.ctx:
            # Saved ctx (e.g. 2000) exceeds demo corpus tokens: W1 shape is
            # tied to ctx so it cannot be clamped in place — fall back to a
            # fresh small-ctx model instead of crashing with IndexError.
            print(f"demo: corpus has {len(ids)} tokens but loaded model "
                  f"needs ctx={model.ctx}; using a fresh model instead")
            raise ValueError("loaded ctx too large for demo corpus")
    except Exception:
        ctx = max(1, min(50, len(ids) - 1))
        model = TinyLM(vocab_size=len(tok.vocab), ctx=ctx, hidden=50, emb=50)
    train(model, ids, steps=100)

    print()
    print(generate(model, tok, input(">>> "), n_tokens=60))
    save_model(model, "tinyLM")


if __name__ == "__main__":
    main()
