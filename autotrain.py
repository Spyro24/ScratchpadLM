"""Auto-Training System: RatingCollector, AutoTrainer, ModelSelector.

Stdlib only. Implements online learning from user feedback.
"""

import json
import time
import threading
import math
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Callable, Tuple
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

    def __init__(self, similarity_threshold: float = 0.2, max_history: int = 1000):
        self.threshold = similarity_threshold
        self.max_history = max_history
        self._seen_hashes: set = set()
        self._seen_prompts: List[str] = []  # For edit distance comparison

    def _edit_distance(self, s1: str, s2: str) -> int:
        """Compute Levenshtein edit distance between two strings."""
        if len(s1) < len(s2):
            s1, s2 = s2, s1
        if len(s2) == 0:
            return len(s1)
        
        previous_row = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        return previous_row[-1]

    def _normalized_distance(self, s1: str, s2: str) -> float:
        """Normalized edit distance (0 = identical, 1 = completely different)."""
        max_len = max(len(s1), len(s2))
        if max_len == 0:
            return 0.0
        return self._edit_distance(s1, s2) / max_len

    def is_duplicate(self, entry: FeedbackEntry) -> bool:
        """Check if entry is a near-duplicate of a previously seen prompt."""
        if entry.prompt_hash in self._seen_hashes:
            return True

        # Check edit distance against recent prompts
        for seen_prompt in self._seen_prompts:
            dist = self._normalized_distance(entry.prompt, seen_prompt)
            if dist < self.threshold:
                return True

        # Add to history
        self._seen_hashes.add(entry.prompt_hash)
        self._seen_prompts.append(entry.prompt)
        
        # Limit history size
        if len(self._seen_prompts) > self.max_history:
            self._seen_prompts = self._seen_prompts[-self.max_history:]
            # Rebuild hash set (simplified - in production would use more efficient structure)
            self._seen_hashes = {hashlib.md5(p.encode()).hexdigest()[:16] for p in self._seen_prompts}
        
        return False

    def reset(self):
        self._seen_hashes.clear()
        self._seen_prompts.clear()


class AutoTrainer:
    """Background worker that auto-trains models on high-rated feedback."""

    def __init__(self, collector: RatingCollector,
                 model_loader_factory: Callable[[], Any],
                 config: Optional[Dict[str, Any]] = None,
                 catalog=None):
        self.collector = collector
        self.model_loader_factory = model_loader_factory
        self.config = config or {}
        self.min_batch = self.config.get("min_batch", DEFAULT_MIN_BATCH)
        self.max_wait = self.config.get("max_wait", DEFAULT_MAX_WAIT)
        self.autotrain_steps = self.config.get("autotrain_steps", DEFAULT_AUTOTRAIN_STEPS)
        self.autotrain_lr = self.config.get("autotrain_lr", DEFAULT_AUTOTRAIN_LR)
        self.rate_limit_minutes = self.config.get("rate_limit_minutes", DEFAULT_RATE_LIMIT_MINUTES)
        self.max_steps_per_day = self.config.get("max_steps_per_day", DEFAULT_MAX_STEPS_PER_DAY)

        # Use provided catalog or fall back to global
        if catalog is None:
            from model_mgmt import MODEL_CATALOG
            self.catalog = MODEL_CATALOG
        else:
            self.catalog = catalog

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

        # Load persisted state
        self._load_state()

    def _load_state(self) -> None:
        """Load persisted autotrain state from disk."""
        state_path = Path(AUTOTRAIN_CONFIG_PATH)
        if state_path.exists():
            try:
                with open(state_path, "r") as f:
                    state = json.load(f)
                self._last_trained = state.get("last_trained", {})
                self._steps_today = defaultdict(int, state.get("steps_today", {}))
                self._last_day = state.get("last_day", time.strftime("%Y-%m-%d"))
                # Restore running state
                if state.get("was_running", False):
                    self.start()
            except Exception:
                pass

    def _save_state(self) -> None:
        """Persist autotrain state to disk."""
        state = {
            "last_trained": self._last_trained,
            "steps_today": dict(self._steps_today),
            "last_day": self._last_day,
            "was_running": self._running,
        }
        try:
            with open(AUTOTRAIN_CONFIG_PATH, "w") as f:
                json.dump(state, f)
        except Exception:
            pass

    def _on_new_feedback(self, entry: FeedbackEntry) -> None:
        """Callback when new feedback is collected."""
        # Update avg_rating metric in catalog
        self._update_avg_rating(entry.model_name)
        
        if self.collector.is_high_rated(entry.rating):
            if not self.deduplicator.is_duplicate(entry):
                self._pending[entry.model_name].append(entry)

    def _update_avg_rating(self, model_name: str) -> None:
        """Calculate and update average rating for a model in the catalog."""
        spec = self.catalog.get(model_name)
        if not spec:
            return
        
        # Get all feedback for this model
        entries = self.collector.get_recent(model_name, limit=10000)
        if not entries:
            return
        
        # Calculate average rating
        total_rating = sum(e.rating for e in entries)
        avg_rating = total_rating / len(entries)
        
        # Update catalog metrics
        self.catalog.update_metrics(model_name, {
            "avg_rating": round(avg_rating, 2),
            "total_ratings": len(entries),
        })

    def _check_rate_limit(self, model_name: str) -> bool:
        """Check if model can be trained (rate limiting)."""
        now = time.time()
        today = time.strftime("%Y-%m-%d")

        # Reset daily counter if new day
        if today != self._last_day:
            self._steps_today.clear()
            self._last_day = today
            self._save_state()

        # Check daily step limit
        if self._steps_today[model_name] >= self.max_steps_per_day:
            return False

        # Check per-model cooldown
        last = self._last_trained.get(model_name, 0)
        if now - last < self.rate_limit_minutes * 60:
            return False

        return True

    def _evaluate_model_loss(self, model, ids: List[int]) -> float:
        """Quick evaluation of model loss on a sample of data."""
        from tinylm import evaluate
        if len(ids) < model.ctx + 1:
            return float('inf')
        # Use a small sample for quick eval
        sample_size = min(50, len(ids) - model.ctx)
        sample_ids = ids[:sample_size + model.ctx]
        try:
            perplexity = evaluate(model, sample_ids)
            return math.log(perplexity) if perplexity > 0 else float('inf')
        except Exception:
            return float('inf')

    def _train_model(self, model_name: str, entries: List[FeedbackEntry]) -> Dict[str, Any]:
        """Train a model on the given feedback entries."""
        import math
        from tinylm import train, sync_vocab, save_model, evaluate
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
        tokenizer_cfg = {"vocab_size": 2000, "min_frequency": 2}
        TOKENIZER_MANAGER.extend(training_text, vocab_size=tokenizer_cfg["vocab_size"],
                                min_frequency=tokenizer_cfg["min_frequency"])
        TOKENIZER_MANAGER.save("bpe_merges.txt")
        sync_vocab(model, len(TOKENIZER_MANAGER.tokenizer.vocab))

        # Encode
        ids = TOKENIZER_MANAGER.tokenizer.encode(training_text)
        if len(ids) <= model.ctx:
            return {"success": False, "error": f"Insufficient tokens: {len(ids)} <= {model.ctx}"}

        # Evaluate loss before training
        loss_before = self._evaluate_model_loss(model, ids)

        # Train
        try:
            train(model, ids, steps=self.autotrain_steps, lr=self.autotrain_lr,
                  batch_size=16, n_negatives=256, workers=1)

            # Evaluate loss after training
            loss_after = self._evaluate_model_loss(model, ids)

            # Save model
            save_model(model, f"models/{model_name}.tlm")
            
            # Update catalog metrics
            spec = MODEL_CATALOG.get(model_name)
            old_train_count = spec.metrics.get("train_count", 0) if spec else 0
            MODEL_CATALOG.update_metrics(model_name, {
                "last_trained": time.time(),
                "train_count": old_train_count + 1,
                "last_loss": loss_after,
                "improvement_delta": loss_before - loss_after if loss_before != float('inf') and loss_after != float('inf') else 0.0,
            })

            self._last_trained[model_name] = time.time()
            self._steps_today[model_name] += self.autotrain_steps
            self._save_state()

            return {
                "success": True, 
                "steps": self.autotrain_steps, 
                "entries": len(entries),
                "loss_before": loss_before,
                "loss_after": loss_after,
                "improvement": loss_before - loss_after if loss_before != float('inf') and loss_after != float('inf') else 0.0
            }
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
            "loss_before": result.get("loss_before"),
            "loss_after": result.get("loss_after"),
            "improvement": result.get("improvement"),
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
        self._save_state()

    def stop(self) -> None:
        """Stop the background auto-trainer."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
            self._thread = None
        self._save_state()

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

    def update_config(self, **kwargs) -> None:
        """Update autotrain configuration."""
        for key, value in kwargs.items():
            if key == "min_batch":
                self.min_batch = value
            elif key == "max_wait":
                self.max_wait = value
            elif key == "autotrain_steps":
                self.autotrain_steps = value
            elif key == "autotrain_lr":
                self.autotrain_lr = value
            elif key == "rate_limit_minutes":
                self.rate_limit_minutes = value
            elif key == "max_steps_per_day":
                self.max_steps_per_day = value
            elif key == "high_threshold":
                self.collector.high_threshold = value
            elif key == "low_threshold":
                self.collector.low_threshold = value
        self.config.update(kwargs)
        self._save_state()

    def handle_low_rated(self, model_name: Optional[str] = None, 
                          since: Optional[float] = None,
                          auto_flag: bool = True) -> List[FeedbackEntry]:
        """Get low-rated entries for review or avoidance training.

        Args:
            model_name: Filter by model (None = all models)
            since: Only entries after this timestamp
            auto_flag: If True, adds 'flagged_for_review' metadata

        Returns:
            List of low-rated FeedbackEntry objects
        """
        entries = self.collector.get_low_rated(model_name, since=since)
        
        if auto_flag:
            for entry in entries:
                if 'flagged_for_review' not in entry.metadata:
                    entry.metadata['flagged_for_review'] = True
                    entry.metadata['flagged_at'] = time.time()
                    # Re-save the entry with flag
                    self._update_feedback_entry(entry)
        
        return entries

    def _update_feedback_entry(self, entry: FeedbackEntry) -> None:
        """Update an existing feedback entry in the JSONL file."""
        # Read all entries, update the matching one, rewrite
        all_entries = []
        if self.collector.feedback_path.exists():
            with open(self.collector.feedback_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        e = FeedbackEntry.from_jsonl(line)
                        if e.prompt_hash == entry.prompt_hash and e.timestamp == entry.timestamp:
                            all_entries.append(entry)
                        else:
                            all_entries.append(e)
                    except Exception:
                        continue
        
        # Rewrite file
        with open(self.collector.feedback_path, "w", encoding="utf-8") as f:
            for e in all_entries:
                f.write(e.to_jsonl() + "\n")

    def train_avoidance(self, model_name: str, entries: List[FeedbackEntry]) -> Dict[str, Any]:
        """Train a model on negative examples (avoidance training).

        This trains the model to avoid generating low-rated responses by using
        the low-rated responses as negative targets.

        Args:
            model_name: Name of the model to train
            entries: Low-rated feedback entries

        Returns:
            Training result dict
        """
        if not entries:
            return {"success": False, "error": "No low-rated entries provided"}

        from tinylm import train, sync_vocab, save_model
        from registry import TOKENIZER_MANAGER
        from model_mgmt import ModelLoader, MODEL_CATALOG

        # Load model
        loader = ModelLoader(MODEL_CATALOG, tokenizer_manager=TOKENIZER_MANAGER)
        model = loader.load_model(model_name)

        # Prepare training data: use prompts with LOW-RATED responses as negative examples
        # We train the model to predict something BETTER than the low-rated response
        # by using the prompt + "better" completion as positive, and prompt + low-rated as negative
        training_text = ""
        for entry in entries:
            # Create contrastive pairs: prompt -> good response vs prompt -> bad response
            # For avoidance, we can use the low-rated response as a negative example
            training_text += f"[AVOID] {entry.prompt} -> {entry.response}\n\n"

        # Extend tokenizer and sync model
        tokenizer_cfg = {"vocab_size": 2000, "min_frequency": 2}
        TOKENIZER_MANAGER.extend(training_text, vocab_size=tokenizer_cfg["vocab_size"],
                                min_frequency=tokenizer_cfg["min_frequency"])
        TOKENIZER_MANAGER.save("bpe_merges.txt")
        sync_vocab(model, len(TOKENIZER_MANAGER.tokenizer.vocab))

        # Encode and train
        ids = TOKENIZER_MANAGER.tokenizer.encode(training_text)
        if len(ids) <= model.ctx:
            return {"success": False, "error": f"Insufficient tokens: {len(ids)} <= {model.ctx}"}

        # Train with slightly different parameters for avoidance
        try:
            train(model, ids, steps=5, lr=0.005,  # Lower LR for avoidance
                  batch_size=8, n_negatives=128, workers=1)

            # Save model with avoidance marker
            save_model(model, f"models/{model_name}.tlm")
            
            MODEL_CATALOG.update_metrics(model_name, {
                "last_avoidance_training": time.time(),
                "avoidance_train_count": MODEL_CATALOG.get(model_name).metrics.get("avoidance_train_count", 0) + 1
            })

            self._save_state()

            return {
                "success": True, 
                "steps": 5, 
                "entries": len(entries),
                "type": "avoidance"
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def flag_for_review(self, model_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get all low-rated entries flagged for human review."""
        entries = self.handle_low_rated(model_name, auto_flag=True)
        
        flagged = []
        for entry in entries:
            if entry.metadata.get('flagged_for_review'):
                flagged.append({
                    "prompt": entry.prompt,
                    "response": entry.response,
                    "model_name": entry.model_name,
                    "rating": entry.rating,
                    "timestamp": entry.timestamp,
                    "metadata": entry.metadata
                })
        
        return flagged

    def get_metrics(self, model_name: str) -> Dict[str, Any]:
        """Get training metrics for a specific model."""
        spec = self.catalog.get(model_name)
        if not spec:
            return {}
        
        metrics = spec.metrics or {}
        
        # Get recent feedback for rating distribution
        entries = self.collector.get_recent(model_name, limit=10000)
        rating_dist = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        for e in entries:
            if 1 <= e.rating <= 5:
                rating_dist[e.rating] += 1
        
        # Calculate high/low rated counts
        high_rated = sum(1 for e in entries if self.collector.is_high_rated(e.rating))
        low_rated = sum(1 for e in entries if self.collector.is_low_rated(e.rating))
        
        return {
            "avg_rating": metrics.get("avg_rating", 0),
            "total_ratings": metrics.get("total_ratings", 0),
            "rating_distribution": rating_dist,
            "high_rated_count": high_rated,
            "low_rated_count": low_rated,
            "train_count": metrics.get("train_count", 0),
            "last_trained": metrics.get("last_trained", 0),
            "last_loss": metrics.get("last_loss"),
            "improvement_delta": metrics.get("improvement_delta", 0),
            "pending_feedback": len(self._pending.get(model_name, [])),
            "high_threshold": self.collector.high_threshold,
            "low_threshold": self.collector.low_threshold,
        }

    def get_all_metrics(self) -> Dict[str, Any]:
        """Get metrics for all models in the catalog."""
        all_metrics = {}
        for spec in self.catalog.list():
            all_metrics[spec.name] = self.get_metrics(spec.name)
        return all_metrics

    def get_system_status(self) -> Dict[str, Any]:
        """Get overall auto-training system status."""
        all_metrics = self.get_all_metrics()
        
        total_ratings = sum(m.get("total_ratings", 0) for m in all_metrics.values())
        total_trains = sum(m.get("train_count", 0) for m in all_metrics.values())
        total_pending = sum(m.get("pending_feedback", 0) for m in all_metrics.values())
        
        # Calculate global avg rating
        all_entries = self.collector.get_recent(limit=10000)
        global_avg_rating = 0
        if all_entries:
            global_avg_rating = round(sum(e.rating for e in all_entries) / len(all_entries), 2)
        
        # Models needing attention (low rated)
        needs_attention = [
            name for name, m in all_metrics.items() 
            if m.get("low_rated_count", 0) > 0
        ]
        
        return {
            "autotrainer_running": self._running,
            "global_avg_rating": global_avg_rating,
            "total_ratings": total_ratings,
            "total_train_sessions": total_trains,
            "total_pending_feedback": total_pending,
            "models_needing_attention": needs_attention,
            "models_count": len(all_metrics),
            "per_model": all_metrics,
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


# =============================================================================
# Training Priority Queue: Scores models by (domain_gap * data_volume * performance_delta)
# =============================================================================

class TrainingPriorityQueue:
    """Priority queue for model training based on domain gaps, data availability, and performance."""
    
    def __init__(self, catalog, collector: RatingCollector, tokenizer_manager=None):
        self.catalog = catalog
        self.collector = collector
        if tokenizer_manager is None:
            from registry import TOKENIZER_MANAGER
            self.tokenizer_manager = TOKENIZER_MANAGER
        else:
            self.tokenizer_manager = tokenizer_manager
        
        self._queue: List[Tuple[float, str, Dict[str, Any]]] = []  # (priority, model_name, metadata)
        self._training_history: Dict[str, List[Dict]] = {}  # model_name -> list of training records
    
    def _calculate_domain_gap(self, model_name: str) -> float:
        """Calculate domain coverage gap for a model."""
        spec = self.catalog.get(model_name)
        if not spec:
            return 1.0  # New model = max gap
        
        # Check how well the model's domains are covered by training data
        domain_keywords = spec.domain_keywords
        if not domain_keywords:
            return 0.5
        
        # Get recent feedback for this model
        entries = self.collector.get_recent(model_name, limit=1000)
        
        # Count keyword coverage in feedback prompts
        covered_keywords = set()
        for entry in entries:
            prompt_lower = entry.prompt.lower()
            for kw in domain_keywords:
                if kw in prompt_lower:
                    covered_keywords.add(kw)
        
        if not domain_keywords:
            return 0.0
        
        gap = 1.0 - (len(covered_keywords) / len(domain_keywords))
        return max(0.0, gap)
    
    def _calculate_data_volume(self, model_name: str) -> float:
        """Calculate available training data volume for a model."""
        entries = self.collector.get_recent(model_name, limit=10000)
        high_rated = [e for e in entries if self.collector.is_high_rated(e.rating)]
        
        # Normalize: 100+ high-rated entries = 1.0
        return min(len(high_rated) / 100.0, 1.0)
    
    def _calculate_performance_delta(self, model_name: str) -> float:
        """Calculate performance improvement potential."""
        spec = self.catalog.get(model_name)
        if not spec:
            return 0.5  # New model has high potential
        
        metrics = spec.metrics or {}
        avg_rating = metrics.get("avg_rating", 3.0)
        train_count = metrics.get("train_count", 0)
        improvement_delta = metrics.get("improvement_delta", 0.0)
        
        # Low rating + few training sessions = high potential
        rating_gap = max(0.0, (4.0 - avg_rating) / 4.0)  # 4.0 is target
        experience_factor = min(train_count / 10.0, 1.0)  # Diminishes with experience
        recent_improvement = max(0.0, improvement_delta)  # Positive improvement reduces priority
        
        # High potential if: low rating, few training sessions, no recent improvement
        potential = rating_gap * (1.0 - experience_factor * 0.5) * (1.0 - min(recent_improvement, 1.0))
        return potential
    
    def score_model(self, model_name: str, data_volume_override: float = None) -> Tuple[float, Dict[str, float]]:
        """Calculate training priority score for a model.
        
        Score = domain_gap * data_volume * performance_delta
        
        Returns:
            (priority_score, component_scores)
        """
        domain_gap = self._calculate_domain_gap(model_name)
        data_volume = data_volume_override if data_volume_override is not None else self._calculate_data_volume(model_name)
        performance_delta = self._calculate_performance_delta(model_name)
        
        priority = domain_gap * data_volume * performance_delta
        
        components = {
            "domain_gap": domain_gap,
            "data_volume": data_volume,
            "performance_delta": performance_delta,
            "priority": priority,
        }
        
        return priority, components
    
    def get_training_priorities(self, max_models: int = 10, 
                                data_volume_overrides: Dict[str, float] = None) -> List[Dict[str, Any]]:
        """Get prioritized list of models for training.
        
        Args:
            max_models: Maximum number of models to return
            data_volume_overrides: Optional dict of model_name -> data_volume (e.g., from new user data)
            
        Returns:
            List of dicts with model_name, priority, components, sorted by priority descending
        """
        data_volume_overrides = data_volume_overrides or {}
        results = []
        
        for spec in self.catalog.list():
            if spec.name == "router":
                continue  # Skip router
            
            priority, components = self.score_model(spec.name, 
                                                   data_volume_overrides.get(spec.name))
            
            results.append({
                "model_name": spec.name,
                "priority": priority,
                "components": components,
                "domains": spec.domain_keywords,
                "param_count": spec.param_count,
            })
        
        # Sort by priority descending
        results.sort(key=lambda x: x["priority"], reverse=True)
        
        return results[:max_models]
    
    def should_train_model(self, model_name: str, threshold: float = 0.1) -> bool:
        """Check if a model should be trained based on priority threshold."""
        priority, _ = self.score_model(model_name)
        return priority >= threshold
    
    def record_training(self, model_name: str, data_volume: float, loss_before: float, 
                        loss_after: float, steps: int) -> None:
        """Record a training session for future priority calculations."""
        if model_name not in self._training_history:
            self._training_history[model_name] = []
        
        self._training_history[model_name].append({
            "timestamp": time.time(),
            "data_volume": data_volume,
            "loss_before": loss_before,
            "loss_after": loss_after,
            "improvement": loss_before - loss_after,
            "steps": steps,
        })
        
        # Keep only last 50 training records
        if len(self._training_history[model_name]) > 50:
            self._training_history[model_name] = self._training_history[model_name][-50:]


def create_autotrain_system(model_loader_factory: Callable[[], Any],
                            config: Optional[Dict[str, Any]] = None,
                            catalog=None) -> tuple:
    """Create a complete auto-training system.

    Args:
        model_loader_factory: Factory function to create ModelLoader
        config: Optional configuration dict
        catalog: Optional ModelCatalog (uses global if not provided)

    Returns:
        (collector, auto_trainer)
    """
    cfg = config or {}
    collector = RatingCollector(
        high_threshold=cfg.get("high_threshold", DEFAULT_HIGH_THRESHOLD),
        low_threshold=cfg.get("low_threshold", DEFAULT_LOW_THRESHOLD),
    )
    auto_trainer = AutoTrainer(collector, model_loader_factory, cfg, catalog)
    return collector, auto_trainer


# =============================================================================
# ModelSelector: Selects the best model for a given input prompt
# =============================================================================

class ModelSelector:
    """Selects the best model for a given input prompt using router + catalog."""

    def __init__(self, router, catalog, tokenizer_manager=None):
        """Initialize the model selector.

        Args:
            router: Router model instance with route() and route_multi() methods
            catalog: ModelCatalog instance
            tokenizer_manager: Optional TokenizerManager (uses global if not provided)
        """
        self.router = router
        self.catalog = catalog
        if tokenizer_manager is None:
            from registry import TOKENIZER_MANAGER
            self.tokenizer_manager = TOKENIZER_MANAGER
        else:
            self.tokenizer_manager = tokenizer_manager

    def select_model_for_input(self, prompt: str, max_candidates: int = 3) -> List[Dict[str, Any]]:
        """Select the best model(s) for a given input prompt.

        Uses the router's routing capabilities combined with catalog metadata
        (domain_keywords, model metrics) to determine the best model.

        Args:
            prompt: Input prompt to route
            max_candidates: Maximum number of model candidates to return

        Returns:
            List of dicts with keys: model, confidence, reasoning
            Sorted by confidence descending
        """
        # Get router's multi-model routing decision
        router_results = self.router.route_multi(
            prompt, self.tokenizer_manager.tokenizer, self.catalog, max_subtasks=max_candidates
        )

        # Enhance with additional scoring
        results = []
        for route in router_results:
            model_name = route["model"]
            base_confidence = route.get("confidence", 0.5)

            # Get model spec from catalog
            spec = self.catalog.get(model_name)
            if spec is None:
                continue

            # Calculate enhanced confidence
            enhanced_confidence = base_confidence

            # Factor 1: Model metrics (avg_rating, train_count)
            metrics = spec.metrics or {}
            avg_rating = metrics.get("avg_rating", 3.0)  # Default neutral
            train_count = metrics.get("train_count", 0)
            last_trained = metrics.get("last_trained", 0)

            # Rating boost (3.0 = neutral, 5.0 = max)
            rating_boost = (avg_rating - 3.0) * 0.1

            # Training experience boost (capped)
            exp_boost = min(train_count * 0.02, 0.2)

            # Recency boost (if trained recently)
            recency_boost = 0.0
            if last_trained > 0:
                days_since = (time.time() - last_trained) / 86400
                if days_since < 7:
                    recency_boost = 0.1 * (1 - days_since / 7)

            # Domain keyword match boost
            prompt_lower = prompt.lower()
            domain_boost = sum(0.1 for kw in spec.domain_keywords if kw in prompt_lower)

            enhanced_confidence += rating_boost + exp_boost + recency_boost + domain_boost
            enhanced_confidence = min(max(enhanced_confidence, 0.0), 1.0)

            results.append({
                "model": model_name,
                "confidence": enhanced_confidence,
                "base_confidence": base_confidence,
                "rating_boost": rating_boost,
                "exp_boost": exp_boost,
                "recency_boost": recency_boost,
                "domain_boost": domain_boost,
                "avg_rating": avg_rating,
                "train_count": train_count,
                "domain_keywords": spec.domain_keywords,
                "reasoning": self._generate_reasoning(route, spec, enhanced_confidence)
            })

        # Sort by enhanced confidence
        results.sort(key=lambda x: x["confidence"], reverse=True)

        # If no results from router, fall back to keyword matching
        if not results:
            results = self._fallback_keyword_match(prompt, max_candidates)

        return results[:max_candidates]

    def _generate_reasoning(self, route: Dict, spec, confidence: float) -> str:
        """Generate human-readable reasoning for the selection."""
        parts = []
        if route.get("confidence", 0) > 0.5:
            parts.append("router confidence high")
        if spec.metrics.get("avg_rating", 0) > 3.5:
            parts.append(f"high avg rating ({spec.metrics['avg_rating']:.1f})")
        if spec.metrics.get("train_count", 0) > 5:
            parts.append(f"well-trained ({spec.metrics['train_count']} sessions)")
        if any(kw in prompt.lower() for kw in spec.domain_keywords for prompt in [prompt]):
            parts.append("domain keyword match")
        return "; ".join(parts) if parts else "default selection"

    def _fallback_keyword_match(self, prompt: str, max_candidates: int) -> List[Dict[str, Any]]:
        """Fallback model selection using simple keyword matching."""
        prompt_lower = prompt.lower()
        scores = []

        for spec in self.catalog.list():
            if spec.name == "router":
                continue
            score = sum(1 for kw in spec.domain_keywords if kw in prompt_lower)
            if score > 0:
                metrics = spec.metrics or {}
                avg_rating = metrics.get("avg_rating", 3.0)
                score += avg_rating * 0.3
                scores.append((spec.name, score))

        scores.sort(key=lambda x: x[1], reverse=True)

        return [
            {
                "model": name,
                "confidence": min(score / 10.0, 1.0),
                "base_confidence": 0.0,
                "rating_boost": 0.0,
                "exp_boost": 0.0,
                "recency_boost": 0.0,
                "domain_boost": score,
                "avg_rating": 3.0,
                "train_count": 0,
                "domain_keywords": [],
                "reasoning": "fallback keyword match"
            }
            for name, score in scores[:max_candidates]
        ]

    def get_best_model(self, prompt: str) -> Optional[str]:
        """Get the single best model name for a prompt."""
        results = self.select_model_for_input(prompt, max_candidates=1)
        return results[0]["model"] if results else None

    def select_and_load(self, prompt: str, model_loader_factory: Callable[[], Any]) -> tuple:
        """Select best model and load it.

        Returns:
            (model_name, model_instance) or (None, None) if no model found
        """
        best_model = self.get_best_model(prompt)
        if best_model is None:
            return None, None

        try:
            loader = model_loader_factory()
            model = loader.load_model(best_model)
            return best_model, model
        except Exception as e:
            return best_model, None


if __name__ == "__main__":
    # Demo
    collector = RatingCollector()
    collector.collect("test prompt", "test response", "tiny-prose", 5, {"source": "demo"})
    collector.collect("another prompt", "another response", "tiny-code", 3, {})
    collector.collect("bad prompt", "bad response", "tiny-prose", 1, {})

    print("High rated:", [e.rating for e in collector.get_high_rated()])
    print("Low rated:", [e.rating for e in collector.get_low_rated()])
    print("All recent:", [(e.model_name, e.rating) for e in collector.get_recent(limit=10)])