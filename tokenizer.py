import re
from collections import Counter, defaultdict
from multiprocessing import Pool, cpu_count


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