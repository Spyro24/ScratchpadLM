import re
import json
import random
from collections import Counter, defaultdict
from multiprocessing import Pool, cpu_count
from typing import List, Dict, Tuple, Optional, Set


PAT = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?[A-Za-z]+| ?[0-9]+| ?[^\sA-Za-z0-9]+|\s+""")


def _count_chunk(chunk: str) -> Counter:
    """Runs in a worker process: count word occurrences in one chunk of text."""
    return Counter(PAT.findall(chunk))


def _split_chunks(text: str, n_chunks: int):
    """Split text into roughly equal pieces at whitespace, so no word is cut in half."""
    size = max(1, len(text) // n_chunks)
    chunks, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        # Move the cut forward to the next whitespace so words stay whole
        while end < len(text) and not text[end].isspace():
            end += 1
        chunks.append(text[start:end])
        start = end
    return chunks


class BPETokenizer:
    """Byte-level BPE tokenizer that learns frequent multi-character patterns."""

    PAT = PAT

    def __init__(self):
        self.merges = {}
        self.vocab = {i: bytes([i]) for i in range(256)}
    
    def extend(self, text, vocab_size, min_frequency=2):
        """Learn new merges from `text`. Existing token IDs are kept. If a merge
        produces bytes that already exist as a token, that token is reused."""
        known = {b: i for i, b in self.vocab.items()}      # bytes -> existing id

        # Re-encode the new text with the merges learned so far
        word_freq = Counter(self.PAT.findall(text))
        words = {}
        for w in word_freq:
            seq = tuple(w.encode("utf-8"))
            for pair, idx in sorted(self.merges.items(), key=lambda kv: kv[1]):
                seq = self._merge_seq(seq, pair, idx)
            words[w] = seq

        pair_counts = Counter()
        pair_words = defaultdict(set)
        for w, seq in words.items():
            f = word_freq[w]
            for pair in zip(seq, seq[1:]):
                pair_counts[pair] += f
                pair_words[pair].add(w)

        next_id = max(self.vocab) + 1
        while len(self.vocab) < vocab_size and pair_counts:
            best, count = pair_counts.most_common(1)[0]
            if count < min_frequency:
                break

            merged = self.vocab[best[0]] + self.vocab[best[1]]
            if merged in known:
                new_id = known[merged]                     # reuse existing token
            else:
                new_id = next_id
                next_id += 1
                self.vocab[new_id] = merged
                known[merged] = new_id

            self.merges[best] = new_id
            for w in list(pair_words.pop(best, ())):
                seq, f = words[w], word_freq[w]
                for pair in zip(seq, seq[1:]):
                    pair_counts[pair] -= f
                    if pair_counts[pair] <= 0:
                        del pair_counts[pair]
                    pair_words[pair].discard(w)
                new_seq = self._merge_seq(seq, best, new_id)
                words[w] = new_seq
                for pair in zip(new_seq, new_seq[1:]):
                    pair_counts[pair] += f
                    pair_words[pair].add(w)
            pair_counts.pop(best, None)
    # ---- training ----------------------------------------------------------

    def train(self, text: str, vocab_size: int, min_frequency: int = 2, workers=None):
        assert vocab_size >= 256, "vocab_size must be at least 256"
        workers = workers or cpu_count()

        # Parallel step: count words in separate processes, then merge the counts
        chunks = _split_chunks(text, workers * 4)
        word_freq = Counter()
        if workers > 1 and len(chunks) > 1:
            with Pool(workers) as pool:
                for counts in pool.imap_unordered(_count_chunk, chunks):
                    word_freq.update(counts)
        else:
            word_freq = _count_chunk(text)

        words = {w: tuple(w.encode("utf-8")) for w in word_freq}

        # Sequential step: each merge depends on the previous one
        pair_counts = Counter()
        pair_words = defaultdict(set)
        for w, seq in words.items():
            f = word_freq[w]
            for pair in zip(seq, seq[1:]):
                pair_counts[pair] += f
                pair_words[pair].add(w)

        next_id = 256
        while next_id < vocab_size and pair_counts:
            best, count = pair_counts.most_common(1)[0]
            if count < min_frequency:
                break

            self.merges[best] = next_id
            self.vocab[next_id] = self.vocab[best[0]] + self.vocab[best[1]]

            for w in list(pair_words.pop(best, ())):
                seq = words[w]
                f = word_freq[w]
                for pair in zip(seq, seq[1:]):
                    pair_counts[pair] -= f
                    if pair_counts[pair] <= 0:
                        del pair_counts[pair]
                    pair_words[pair].discard(w)

                new_seq = self._merge_seq(seq, best, next_id)
                words[w] = new_seq
                for pair in zip(new_seq, new_seq[1:]):
                    pair_counts[pair] += f
                    pair_words[pair].add(w)

            pair_counts.pop(best, None)
            next_id += 1

    @staticmethod
    def _merge_seq(seq, pair, new_id):
        out, i = [], 0
        while i < len(seq):
            if i < len(seq) - 1 and (seq[i], seq[i + 1]) == pair:
                out.append(new_id)
                i += 2
            else:
                out.append(seq[i])
                i += 1
        return tuple(out)

    # ---- encoding / decoding and persistence are unchanged -----------------

    def encode(self, text: str):
        ids = []
        for word in PAT.findall(text):
            seq = tuple(word.encode("utf-8"))
            while len(seq) >= 2:
                candidates = [p for p in zip(seq, seq[1:]) if p in self.merges]
                if not candidates:
                    break
                best = min(candidates, key=lambda p: self.merges[p])
                seq = self._merge_seq(seq, best, self.merges[best])
            ids.extend(seq)
        return ids

    def decode(self, ids):
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")

    def tokens(self, text: str):
        return [self.vocab[i].decode("utf-8", errors="replace") for i in self.encode(text)]

    def learned_patterns(self, min_len=2):
        return [self.vocab[i].decode("utf-8", errors="replace")
                for i in sorted(self.merges.values())
                if len(self.vocab[i]) >= min_len]

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            for (a, b), idx in self.merges.items():
                f.write(f"{a} {b} {idx}\n")

    def load(self, path):
        self.merges.clear()
        self.vocab = {i: bytes([i]) for i in range(256)}
        with open(path, encoding="utf-8") as f:
            for line in f:
                a, b, idx = map(int, line.split())
                self.merges[(a, b)] = idx
                self.vocab[idx] = self.vocab[a] + self.vocab[b]

    # ---- word frequency cache -------------------------------------------------

    def save_word_freq(self, path: str):
        """Save word frequency cache to file for faster re-training."""
        # Re-compute word frequencies from current vocabulary
        # This is an approximation since we don't store original corpus
        word_freq = Counter()
        for word_bytes in self.vocab.values():
            if len(word_bytes) > 1:  # Only multi-byte tokens (learned merges)
                try:
                    word = word_bytes.decode("utf-8")
                    word_freq[word] = 1  # Placeholder frequency
                except UnicodeDecodeError:
                    pass
        
        # Save as JSON
        data = {word: count for word, count in word_freq.items()}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)

    def load_word_freq(self, path: str) -> Counter:
        """Load word frequency cache from file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return Counter(data)

    def train_with_cache(self, text: str, vocab_size: int, min_frequency: int = 2, 
                         workers=None, word_freq_cache: Optional[Counter] = None):
        """Train tokenizer using cached word frequencies if available."""
        if word_freq_cache is not None:
            # Use cached frequencies for words present in cache
            word_freq = word_freq_cache.copy()
            # Update with new text frequencies
            new_freq = Counter(self.PAT.findall(text))
            word_freq.update(new_freq)
        else:
            # Standard training path
            return self.train(text, vocab_size, min_frequency, workers)
        
        # Continue with sequential merge learning using cached frequencies
        words = {w: tuple(w.encode("utf-8")) for w in word_freq}
        
        # Apply existing merges to cached words
        for w, seq in words.items():
            for pair, idx in sorted(self.merges.items(), key=lambda kv: kv[1]):
                seq = self._merge_seq(seq, pair, idx)
            words[w] = seq
        
        # Sequential step: each merge depends on the previous one
        pair_counts = Counter()
        pair_words = defaultdict(set)
        for w, seq in words.items():
            f = word_freq[w]
            for pair in zip(seq, seq[1:]):
                pair_counts[pair] += f
                pair_words[pair].add(w)

        next_id = max(self.vocab) + 1
        while next_id < vocab_size and pair_counts:
            best, count = pair_counts.most_common(1)[0]
            if count < min_frequency:
                break

            merged = self.vocab[best[0]] + self.vocab[best[1]]
            if merged in {v for v in self.vocab.values()}:
                # Find existing ID
                for idx, val in self.vocab.items():
                    if val == merged:
                        new_id = idx
                        break
            else:
                new_id = next_id
                next_id += 1
                self.vocab[new_id] = merged

            self.merges[best] = new_id
            for w in list(pair_words.pop(best, ())):
                seq, f = words[w], word_freq[w]
                for pair in zip(seq, seq[1:]):
                    pair_counts[pair] -= f
                    if pair_counts[pair] <= 0:
                        del pair_counts[pair]
                    pair_words[pair].discard(w)
                new_seq = self._merge_seq(seq, best, new_id)
                words[w] = new_seq
                for pair in zip(new_seq, new_seq[1:]):
                    pair_counts[pair] += f
                    pair_words[pair].add(w)
            pair_counts.pop(best, None)


if __name__ == "__main__":
    # Multiprocessing needs this guard on Windows and macOS
    corpus = (
        "the quick brown fox jumps over the lazy dog. "
        "the dog was not interested in the fox. "
        "a small model learns patterns from text and from the words in text. "
    ) * 20

    tok = BPETokenizer()
    tok.train(corpus, vocab_size=400)
    print(tok.learned_patterns())


# =============================================================================
# Domain Detection for Auto Model Naming
# =============================================================================

DOMAIN_KEYWORDS = {
    "code": [
        "def ", "class ", "import ", "from ", "function", "return ", "if __name__",
        "console.log", "println", "System.out", "public ", "private ", "void ",
        "int ", "string ", "bool ", "const ", "let ", "var ", "async ", "await ",
        "=>", "()", "{}", "[]", "npm ", "pip ", "cargo ", "go ", "rust ", "python",
        "javascript", "typescript", "java", "c++", "c#", "swift", "kotlin",
        "github", "git ", "commit", "push", "pull", "merge", "branch",
        "def ", "lambda", "yield ", "with ", "async def", "await ",
    ],
    "reason": [
        "therefore", "thus", "hence", "because", "since", "implies", "iff",
        "proof", "theorem", "lemma", "corollary", "axiom", "proposition",
        "solve", "calculate", "derive", "integral", "derivative", "limit",
        "matrix", "vector", "eigenvalue", "probability", "statistics",
        "hypothesis", "assume", "suppose", "let ", "given ", "prove ",
        "math", "logic", "reason", "deduce", "infer", "conclude",
    ],
    "prose": [
        "once upon", "story", "chapter", "paragraph", "sentence", "narrative",
        "character", "plot", "setting", "dialogue", "author", "novel", "book",
        "write", "written", "writing", "article", "essay", "blog", "post",
        "summary", "summarize", "explain", "describe", "tell me", "story about",
    ],
    "sql": [
        "select ", "from ", "where ", "join ", "group by", "order by",
        "insert ", "update ", "delete ", "create table", "alter table",
        "drop table", "index ", "primary key", "foreign key", "inner join",
        "left join", "right join", "union ", "distinct ", "count(",
        "sum(", "avg(", "max(", "min(", "having ", "sql", "database",
    ],
    "shell": [
        "bash", "shell", "command", "terminal", "sudo ", "apt ", "yum ",
        "brew ", "npm ", "pip ", "docker ", "kubectl ", "git ",
        "ls ", "cd ", "mkdir ", "rm ", "cp ", "mv ", "cat ", "grep ",
        "awk ", "sed ", "curl ", "wget ", "ssh ", "scp ", "rsync ",
        "chmod ", "chown ", "systemctl ", "service ", "ps ", "kill ",
    ],
    "docs": [
        "documentation", "manual", "guide", "tutorial", "readme", "wiki",
        "api reference", "specification", "spec ", "overview", "getting started",
        "installation", "configuration", "usage", "example", "parameter",
        "return value", "exception", "throws", "deprecated", "version",
    ],
    "data": [
        "pandas", "numpy", "dataframe", "series", "csv", "json", "parquet",
        "plot", "chart", "graph", "visualize", "matplotlib", "seaborn",
        "analysis", "statistics", "mean", "median", "std", "correlation",
        "regression", "classification", "clustering", "pca", "model",
        "train", "test", "predict", "accuracy", "precision", "recall",
    ],
    "web": [
        "http", "https", "api", "rest", "graphql", "endpoint", "request",
        "response", "json", "xml", "html", "css", "javascript", "frontend",
        "backend", "server", "client", "authentication", "authorization",
        "token", "oauth", "jwt", "cookie", "session", "cors",
    ],
    "ml": [
        "machine learning", "deep learning", "neural network", "tensorflow",
        "pytorch", "keras", "sklearn", "model", "train", "epoch", "batch",
        "loss", "accuracy", "gradient", "optimizer", "backprop", "layer",
        "convolution", "pooling", "dropout", "activation", "relu", "softmax",
        "transformer", "attention", "embedding", "tokenizer", "dataset",
    ],
}


def detect_domains(text: str, top_k: int = 3) -> List[str]:
    """Detect domains from text content.
    
    Args:
        text: Input text to analyze
        top_k: Number of top domains to return
        
    Returns:
        List of domain names sorted by relevance
    """
    text_lower = text.lower()
    scores = {}
    
    for domain, keywords in DOMAIN_KEYWORDS.items():
        score = 0
        for kw in keywords:
            count = text_lower.count(kw.lower())
            if count > 0:
                # Weight longer keywords more
                score += count * len(kw)
        if score > 0:
            scores[domain] = score
    
    # Sort by score descending
    sorted_domains = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [domain for domain, _ in sorted_domains[:top_k]]


def generate_model_name(domains: List[str], existing_names: Set[str] = None) -> str:
    """Generate a model name from detected domains.
    
    Args:
        domains: List of detected domain names
        existing_names: Set of existing model names to avoid conflicts
        
    Returns:
        Generated model name (e.g., "tiny-code", "tiny-code-reason")
    """
    if not domains:
        return "tiny-custom"
    
    if existing_names is None:
        existing_names = set()
    
    # Use first domain as primary, add others if available
    if len(domains) == 1:
        base = f"tiny-{domains[0]}"
    else:
        base = f"tiny-{domains[0]}-{domains[1]}"
    
    # Ensure uniqueness
    name = base
    counter = 1
    while name in existing_names:
        counter += 1
        name = f"{base}-{counter}"
    
    return name


def analyze_training_data(text: str, existing_names: Set[str] = None) -> Dict[str, any]:
    """Analyze training data and suggest model creation.
    
    Args:
        text: Training text to analyze
        existing_names: Set of existing model names
        
    Returns:
        Dict with keys: domains, suggested_name, confidence, train_or_create
    """
    domains = detect_domains(text)
    suggested_name = generate_model_name(domains, existing_names)
    
    # Determine confidence based on keyword matches
    text_lower = text.lower()
    total_matches = 0
    for domain in domains:
        for kw in DOMAIN_KEYWORDS.get(domain, []):
            total_matches += text_lower.count(kw.lower())
    
    confidence = min(total_matches / 100.0, 1.0)  # Normalize
    
    # Determine if we should fine-tune existing or create new
    train_or_create = "create"
    if existing_names:
        # Check if any existing model matches the primary domain
        primary_domain = domains[0] if domains else ""
        for existing in existing_names:
            if primary_domain in existing.lower():
                train_or_create = "fine-tune"
                break
    
    return {
        "domains": domains,
        "suggested_name": suggested_name,
        "confidence": confidence,
        "train_or_create": train_or_create,
        "primary_domain": domains[0] if domains else "unknown",
    }