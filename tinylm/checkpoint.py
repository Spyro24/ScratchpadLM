"""Training checkpoints: save/load/resume + flat param helpers (stdlib only)."""
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

@dataclass
class Checkpoint:
    """Training checkpoint with model state, optimizer state, and metadata."""
    step: int
    model_state: list  # flat list of parameter values
    config: dict
    loss: float
    val_loss: Optional[float] = None
    lr: float = 0.0

    def to_dict(self) -> dict:
        import json
        return {
            "step": self.step,
            "model_state": self.model_state,
            "config": self.config,
            "loss": self.loss,
            "val_loss": self.val_loss,
            "lr": self.lr,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Checkpoint":
        return cls(
            step=data["step"],
            model_state=data["model_state"],
            config=data["config"],
            loss=data["loss"],
            val_loss=data.get("val_loss"),
            lr=data.get("lr", 0.0),
        )


def save_checkpoint(checkpoint: Checkpoint, path: str):
    """Save checkpoint to JSON file."""
    import json
    from pathlib import Path
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(checkpoint.to_dict(), f)


def load_checkpoint(path: str) -> Checkpoint:
    """Load checkpoint from JSON file."""
    import json
    with open(path, "r") as f:
        data = json.load(f)
    return Checkpoint.from_dict(data)


def resume_from_checkpoint(model, checkpoint: Checkpoint):
    """Restore model from checkpoint."""
    for p, val in zip(model.params, checkpoint.model_state):
        p.data = val


def _flat_params(model):
    return [p.data for p in model.params]


def _set_flat_params(model, flat):
    for p, x in zip(model.params, flat):
        p.data = x

