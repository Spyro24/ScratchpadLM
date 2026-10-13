"""Per-model training stats with monotonic best-loss enforcement (stdlib only).

`TrainingStats` persists one record per model (`training_stats.json`):
best/last loss, run/step totals. `train_with_stats()` wraps `train()` so a
model can only decrease its loss across runs: when a run fails to beat the
stored best, the best weights (saved as 4-bit `.tlm` on improvement) are
restored in place and the model file is left for the caller to skip.
"""
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, Optional

STATS_PATH = "training_stats.json"
BEST_DIR = "best_models"


def _safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", name or "model")


class TrainingStats:
    """JSON-backed per-model training records."""

    def __init__(self, path: str = STATS_PATH):
        self.path = Path(path)
        self._records: Dict[str, Dict[str, Any]] = {}
        self._load()

    def _load(self) -> None:
        if self.path.exists():
            try:
                data = json.loads(self.path.read_text(encoding="utf-8"))
            except Exception:
                data = {}
            if isinstance(data, dict):
                self._records = data

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._records, indent=2),
                             encoding="utf-8")

    def get(self, name: str) -> Dict[str, Any]:
        """Record for `name` ({best_loss, last_loss, runs, steps, updated_at})."""
        return self._records.get(_safe_name(name), {})

    def best_loss(self, name: str) -> Optional[float]:
        """Smallest loss ever recorded for `name`, or None."""
        return self.get(name).get("best_loss")

    def record(self, name: str, best_loss: Optional[float],
               last_loss: Optional[float], steps_done: int = 0) -> Dict[str, Any]:
        """Merge one run: best tracks the minimum, totals accumulate."""
        key = _safe_name(name)
        rec = self._records.get(key, {})
        prev_best = rec.get("best_loss")
        if best_loss is not None and (prev_best is None or best_loss < prev_best):
            rec["best_loss"] = best_loss
        if last_loss is not None:
            rec["last_loss"] = last_loss
        rec["runs"] = rec.get("runs", 0) + 1
        rec["steps"] = rec.get("steps", 0) + (steps_done or 0)
        rec["updated_at"] = time.time()
        self._records[key] = rec
        self._save()
        return rec


def best_path(name: str, best_dir: str = BEST_DIR) -> str:
    """Filesystem path of the persisted best weights for `name`."""
    return str(Path(best_dir) / f"best_{_safe_name(name)}.tlm")


def apply_stats(name: str, model, result: Dict[str, Any], stats=None,
                stats_path: str = STATS_PATH, best_dir: str = BEST_DIR) -> Dict[str, Any]:
    """Enforce a non-increasing best loss for `name` given a train result.

    Works with any training path (`train()`, `train_parallel()`, ...): on
    improvement the weights are snapshotted to `best_<name>.tlm` and the
    stats record updated; otherwise the persisted best weights are restored
    in place (the model never regresses) and `reverted` is True. Returns
    `{improved, reverted, prev_best, best_loss}`.
    """
    from tinylm.serialization import load_model, save_model

    store = stats if stats is not None else TrainingStats(stats_path)
    prev_best = store.best_loss(name)
    target = best_path(name, best_dir)

    new_best = result.get("best_loss")
    improved = (
        new_best is not None
        and (prev_best is None or new_best < prev_best)
    )
    reverted = False
    if improved:
        Path(target).parent.mkdir(parents=True, exist_ok=True)
        save_model(model, target)
    elif prev_best is not None and Path(target).exists():
        best_model = load_model(target)
        for p, q in zip(model.params, best_model.params):
            p.data = q.data
        reverted = True
    store.record(name, result.get("best_loss"), result.get("final_loss"),
                 result.get("steps_done", 0) or 0)
    return {
        "improved": improved,
        "reverted": reverted,
        "prev_best": prev_best,
        "best_loss": store.best_loss(name),
    }


def train_with_stats(name: str, model, ids, stats=None,
                     stats_path: str = STATS_PATH, best_dir: str = BEST_DIR,
                     **train_kwargs) -> Dict[str, Any]:
    """Run `train()` and enforce a non-increasing best loss for `name`.

    Returns `{result, improved, reverted, prev_best, best_loss}` (see
    `apply_stats()`).
    """
    from tinylm.training import train

    result = train(model, ids, **train_kwargs)
    info = apply_stats(name, model, result, stats=stats,
                       stats_path=stats_path, best_dir=best_dir)
    info["result"] = result
    return info
