"""TinyLM: embedding -> concat context -> tanh hidden -> softmax."""
import random
from typing import List

from tinylm.autograd import Value, matvec_dot, softmax_cross_entropy

# ---- model ----------------------------------------------------------------

class TinyLM:
    """Predicts the next token from the previous `ctx` tokens.

    Architecture: token embeddings -> concatenate -> tanh hidden layer -> softmax.
    """

    def __init__(self, vocab_size, ctx=3, emb=8, hidden=32, seed=0):
        rng = random.Random(seed)
        self.ctx = ctx

        def matrix(rows, cols, scale=0.1):
            return [[Value(rng.gauss(0.0, scale)) for _ in range(cols)] for _ in range(rows)]

        self.C = matrix(vocab_size, emb)          # embedding table
        self.W1 = matrix(hidden, ctx * emb)
        self.b1 = [Value(0.0) for _ in range(hidden)]
        self.W2 = matrix(vocab_size, hidden)
        self.b2 = [Value(0.0) for _ in range(vocab_size)]

        self.params = (
            [p for row in self.C for p in row]
            + [p for row in self.W1 for p in row] + self.b1
            + [p for row in self.W2 for p in row] + self.b2
        )

    def get_config(self) -> dict:
        """Return model configuration for serialization."""
        return {
            "arch": "TinyLM",
            "vocab_size": len(self.C),
            "ctx": self.ctx,
            "emb": len(self.C[0]),
            "hidden": len(self.b1),
        }

    def get_tensors(self) -> dict:
        """Return model tensors as {name: List[List[Value]]} for serialization."""
        return {
            "C": self.C,
            "W1": self.W1,
            "b1": [self.b1],  # wrap bias as 1-row matrix
            "W2": self.W2,
            "b2": [self.b2],
        }

    def _embed_context(self, context):
        """Concatenate embeddings for context tokens. Returns list of Values."""
        x = []
        for tok in context:
            x.extend(self.C[tok])
        return x

    def logits(self, context):
        x = self._embed_context(context)
        h = matvec_dot(self.W1, x, self.b1)
        for i in range(len(h)):
            h[i] = h[i].tanh()
        return matvec_dot(self.W2, h, self.b2)

    def loss(self, context, target):
        return softmax_cross_entropy(self.logits(context), target)
    
    def sampled_loss(self, context, target, n_negatives, vocab_size, negatives=None):
        """Cross-entropy over the target plus random negatives only.
        Computes just those output rows instead of the whole vocabulary.
        
        Args:
            context: List of token IDs for context
            target: Target token ID
            n_negatives: Number of negative samples
            vocab_size: Vocabulary size
            negatives: Optional pre-sampled negative token IDs
        """
        x = self._embed_context(context)
        h = matvec_dot(self.W1, x, self.b1)
        for i in range(len(h)):
            h[i] = h[i].tanh()

        if negatives is not None:
            candidates = [target] + negatives
        else:
            candidates = [target] + [random.randrange(vocab_size) for _ in range(n_negatives)]
        W2_sub = [self.W2[c] for c in candidates]
        b2_sub = [self.b2[c] for c in candidates]
        logits = matvec_dot(W2_sub, h, b2_sub)
        return softmax_cross_entropy(logits, 0)

