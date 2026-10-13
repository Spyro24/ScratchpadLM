"""Auto-Training System: RatingCollector, AutoTrainer, ModelSelector.

Stdlib only. Implements online learning from user feedback.
"""

import json
import time
import threading
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Callable
from collections import defaultdict
import hashlib


FEEDBACK_PATH = "feedback.jsonl"
AUTOTRAIN_LOG_PATH = "autotrain.log"
AUTOTRAIN_CONFIG_PATH = "autotrain_config.json"


# Default configuration
DEFAULT_HIGH_THRESHOLD = 4
DEFAULT_LOW_THRESHOLD = 2
DEFAULT_MIN_BATCH = 8
DEFAULT_MAX_WAIT = 60  # seconds
DEFAULT_AUTOTRAIN_STEPS = 10
DEFAULT_AUTOTRAIN_LR = 0.01
DEFAULT_RATE_LIMIT_MINUTES = 5
DEFAULT_MAX_STEPS_PER_DAY = 100


@dataclass
class FeedbackEntry:
    """A single feedback entry from user rating."""
    prompt: str
    response: str
    model_name: str
    rating: int  # 1-5
    metadata: Dict[str, Any]
    timestamp: float = 0.0
    prompt_hash: str = ""

    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()
        if not self.prompt_hash:
            self.prompt_hash = hashlib.md5(self.prompt.encode()).hexdigest()[:16]

    def to_jsonl(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_jsonl(cls, line: str) -> "FeedbackEntry":
        data = json.loads(line)
        return cls(**data)


class RatingCollector:
    """Collects and stores user feedback ratings."""

    def __init__(self, feedback_path: str = FEEDBACK_PATH,
                 high_threshold: int = DEFAULT_HIGH_THRESHOLD,
                 low_threshold: int = DEFAULT_LOW_THRESHOLD):
        self.feedback_path = Path(feedback_path)
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self._lock = threading.Lock()
        self._callbacks: List[Callable[[FeedbackEntry], None]] = []

    def collect(self, prompt: str, response: str, model_name: str,
                rating: int, metadata: Optional[Dict[str, Any]] = None) -> FeedbackEntry:
        """Store a feedback entry.

        Args:
            prompt: The input prompt
            response: The model's response
            model_name: Name of the model that generated the response
            rating: User rating (1-5)
            metadata: Optional additional metadata

        Returns:
            The created FeedbackEntry
        """
        if not 1 <= rating <= 5:
            raise ValueError("Rating must be between 1 and 5")

        entry = FeedbackEntry(
            prompt=prompt,
            response=response,
            model_name=model_name,
            rating=rating,
            metadata=metadata or {}
        )

        with self._lock:
            # Append to JSONL file
            with open(self.feedback_path, "a", encoding="utf-8") as f:
                f.write(entry.to_jsonl() + "\n")

        # Trigger callbacks
        for cb in self._callbacks:
            try:
                cb(entry)
            except Exception:
                pass  # Don't let callback errors break collection

        return entry

    def register_callback(self, callback: Callable[[FeedbackEntry], None]) -> None:
        """Register a callback to be called on each new feedback entry."""
        self._callbacks.append(callback)

    def get_recent(self, model_name: Optional[str] = None, limit: int = 100) -> List[FeedbackEntry]:
        """Get recent feedback entries, optionally filtered by model."""
        entries = []
        if not self.feedback_path.exists():
            return entries

        with open(self.feedback_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = FeedbackEntry.from_jsonl(line)
                    if model_name is None or entry.model_name == model_name:
                        entries.append(entry)
                except Exception:
                    continue

        # Return most recent first
        entries.sort(key=lambda e: e.timestamp, reverse=True)
        return entries[:limit]

    def get_high_rated(self, model_name: Optional[str] = None,
                       since: Optional[float] = None) -> List[FeedbackEntry]:
        """Get entries with rating >= high_threshold."""
        entries = self.get_recent(model_name, limit=10000)
        result = [e for e in entries if e.rating >= self.high_threshold]
        if since:
            result = [e for e in result if e.timestamp >= since]
        return result

    def get_low_rated(self, model_name: Optional[str] = None,
                      since: Optional[float] = None) -> List[FeedbackEntry]:
        """Get entries with rating <= low_threshold."""
        entries = self.get_recent(model_name, limit=10000)
        result = [e for e in entries if e.rating <= self.low_threshold]
        if since:
            result = [e for e in result if e.timestamp >= since]
        return result

    def is_high_rated(self, rating: int) -> bool:
        return rating >= self.high_threshold

    def is_low_rated(self, rating: int) -> bool:
        return rating <= self.low_threshold


class Deduplicator:
    """Deduplicates feedback entries based on prompt similarity."""

    def __init__(self, similarity_threshold: float = 0.2):
        self.threshold = similarity_threshold
        self._seen_hashes: set = set()

    def is_duplicate(self, entry: FeedbackEntry) -> bool:
        """Check if entry is a near-duplicate of a previously seen prompt."""
        if entry.prompt_hash in self._seen_hashes:
            return True

        # For more sophisticated dedup, could compute edit distance
        # For now, just use hash-based exact match
        self._seen_hashes.add(entry.prompt_hash)
        return False

    def reset(self):
        self._seen_hashes.clear()


class AutoTrainer:
    """Background worker that auto-trains models on high-rated feedback."""

    def __init__(self, collector: RatingCollector,
                 model_loader_factory: Callable[[], Any],
                 config: Optional[Dict[str, Any]] = None):
        self.collector = collector
        self.model_loader_factory = model_loader_factory
        self.config = config or {}
        self.min_batch = self.config.get("min_batch", DEFAULT_MIN_BATCH)
        self.max_wait = self.config.get("max_wait", DEFAULT_MAX_WAIT)
        self.autotrain_steps = self.config.get("autotrain_steps", DEFAULT_AUTOTRAIN_STEPS)
        self.autotrain_lr = self.config.get("autotrain_lr", DEFAULT_AUTOTRAIN_LR)
        self.rate_limit_minutes = self.config.get("rate_limit_minutes", DEFAULT_RATE_LIMIT_MINUTES)
        self.max_steps_per_day = self.config.get("max_steps_per_day", DEFAULT_MAX_STEPS_PER_DAY)

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._last_trained: Dict[str, float] = {}  # model_name -> timestamp
        self._steps_today: Dict[str, int] = defaultdict(int)
        self._last_day: str = time.strftime("%Y-%m-%d")
        self.deduplicator = Deduplicator()

        # Register callback for new feedback
        self.collector.register_callback(self._on_new_feedback)

        # Pending feedback by model
        self._pending: Dict[str, List[FeedbackEntry]] = defaultdict(list)

    def _on_new_feedback(self, entry: FeedbackEntry) -> None:
        """Callback when new feedback is collected."""
        if self.collector.is_high_rated(entry.rating):
            if not self.deduplicator.is_duplicate(entry):
                self._pending[entry.model_name].append(entry)

    def _check_rate_limit(self, model_name: str) -> bool:
        """Check if model can be trained (rate limiting)."""
        now = time.time()
        today = time.strftime("%Y-%m-%d")

        # Reset daily counter if new day
        if today != self._last_day:
            self._steps_today.clear()
            self._last_day = today

        # Check daily step limit
        if self._steps_today[model_name] >= self.max_steps_per_day:
            return False

        # Check per-model cooldown
        last = self._last_trained.get(model_name, 0)
        if now - last < self.rate_limit_minutes * 60:
            return False

        return True

    def _train_model(self, model_name: str, entries: List[FeedbackEntry]) -> Dict[str, Any]:
        """Train a model on the given feedback entries."""
        from ai import train, sync_vocab, save_model
        from tokenizer import BPETokenizer
        from registry import TOKENIZER_MANAGER
        from model_mgmt import ModelLoader, MODEL_CATALOG

        # Load model
        loader = ModelLoader(MODEL_CATALOG, tokenizer_manager=TOKENIZER_MANAGER)
        model = loader.load_model(model_name)

        # Prepare training data
        training_text = ""
        for entry in entries:
            training_text += entry.prompt + "\n" + entry.response + "\n\n"

        # Extend tokenizer and sync model
        tokenizer_cfg = {"vocab_size": 2000, "min_frequency": 2}  # Could come from config
        TOKENIZER_MANAGER.extend(training_text, vocab_size=tokenizer_cfg["vocab_size"],
                                min_frequency=tokenizer_cfg["min_frequency"])
        TOKENIZER_MANAGER.save("bpe_merges.txt")
        sync_vocab(model, len(TOKENIZER_MANAGER.tokenizer.vocab))

        # Encode and train
        ids = TOKENIZER_MANAGER.tokenizer.encode(training_text)
        if len(ids) <= model.ctx:
            return {"success": False, "error": f"Insufficient tokens: {len(ids)} <= {model.ctx}"}

        # Train
        loss_before = None
        try:
            # Quick eval before training (optional)
            pass

            train(model, ids, steps=self.autotrain_steps, lr=self.autotrain_lr,
                  batch_size=16, n_negatives=256, workers=1)

            # Save model
            save_model(model, f"models/{model_name}.tlm")
            MODEL_CATALOG.update_metrics(model_name, {
                "last_trained": time.time(),
                "train_count": MODEL_CATALOG.get(model_name).metrics.get("train_count", 0) + 1
            })

            self._last_trained[model_name] = time.time()
            self._steps_today[model_name] += self.autotrain_steps

            return {"success": True, "steps": self.autotrain_steps, "entries": len(entries)}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _worker_loop(self) -> None:
        """Main background worker loop."""
        while self._running:
            # Check each model's pending queue
            for model_name, entries in list(self._pending.items()):
                if len(entries) >= self.min_batch:
                    # Check rate limit
                    if self._check_rate_limit(model_name):
                        # Train on batch
                        batch = entries[:self.min_batch]
                        self._pending[model_name] = entries[self.min_batch:]

                        result = self._train_model(model_name, batch)

                        # Log result
                        self._log_autotrain(model_name, batch, result)

            time.sleep(1)  # Check every second

    def _log_autotrain(self, model_name: str, entries: List[FeedbackEntry], result: Dict[str, Any]) -> None:
        """Log autotrain result."""
        log_entry = {
            "timestamp": time.time(),
            "model": model_name,
            "batch_size": len(entries),
            "steps": result.get("steps", 0),
            "success": result.get("success", False),
            "error": result.get("error"),
        }
        try:
            with open(AUTOTRAIN_LOG_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry) + "\n")
        except Exception:
            pass

    def start(self) -> None:
        """Start the background auto-trainer."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the background auto-trainer."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
            self._thread = None

    def status(self) -> Dict[str, Any]:
        """Get auto-trainer status."""
        return {
            "running": self._running,
            "pending_by_model": {k: len(v) for k, v in self._pending.items()},
            "last_trained": self._last_trained,
            "steps_today": dict(self._steps_today),
            "config": {
                "high_threshold": self.collector.high_threshold,
                "low_threshold": self.collector.low_threshold,
                "min_batch": self.min_batch,
                "max_wait": self.max_wait,
                "autotrain_steps": self.autotrain_steps,
                "autotrain_lr": self.autotrain_lr,
                "rate_limit_minutes": self.rate_limit_minutes,
                "max_steps_per_day": self.max_steps_per_day,
            }
        }


def create_autotrain_system(model_loader_factory: Callable[[], Any],
                            config: Optional[Dict[str, Any]] = None) -> tuple:
    """Create a complete auto-training system.

    Returns:
        (collector, auto_trainer)
    """
    cfg = config or {}
    collector = RatingCollector(
        high_threshold=cfg.get("high_threshold", DEFAULT_HIGH_THRESHOLD),
        low_threshold=cfg.get("low_threshold", DEFAULT_LOW_THRESHOLD),
    )
    auto_trainer = AutoTrainer(collector, model_loader_factory, cfg)
    return collector, auto_trainer


if __name__ == "__main__":
    # Demo
    collector = RatingCollector()
    collector.collect("test prompt", "test response", "tiny-prose", 5, {"source": "demo"})
    collector.collect("another prompt", "another response", "tiny-code", 3, {})
    collector.collect("bad prompt", "bad response", "tiny-prose", 1, {})

    print("High rated:", [e.rating for e in collector.get_high_rated()])
    print("Low rated:", [e.rating for e in collector.get_low_rated()])
    print("All recent:", [(e.model_name, e.rating) for e in collector.get_recent(limit=10)])