"""Train-from-scratch demo (run with ``python3 -m tinylm``)."""
import random

from tokenizer import BPETokenizer

from tinylm.generation import generate
from tinylm.models_tiny import TinyLM
from tinylm.serialization import load_model, save_model
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
    except:
        model = TinyLM(vocab_size=len(tok.vocab), ctx=50, hidden=50, emb=50)
    train(model, ids, steps=100)

    print()
    print(generate(model, tok, input(">>> "), n_tokens=60))
    save_model(model, "tinyLM")


if __name__ == "__main__":
    main()
