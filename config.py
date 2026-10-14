"""Config system: YAML/JSON loading, validation, defaults. Stdlib only."""
import json
import os
from typing import Any, Dict, List, Optional, Union
from pathlib import Path


class ConfigError(Exception):
    """Raised when config validation fails."""
    pass


def load_yaml(text: str) -> Dict[str, Any]:
    """Minimal YAML subset parser (stdlib only).
    Supports: key: value, nested dicts, lists, strings, numbers, booleans, null.
    """
    result = {}
    # Stack of (indent, container) where container is the current dict or list
    stack = [(-1, result)]
    lines = text.splitlines()

    for line_num, line in enumerate(lines, 1):
        stripped = line.rstrip()
        if not stripped or stripped.startswith('#'):
            continue

        indent = len(stripped) - len(stripped.lstrip())
        key_value = stripped.lstrip()

        # Handle list items
        if key_value.startswith('- '):
            item = key_value[2:].strip()
            # Find the parent list
            while stack and not isinstance(stack[-1][1], list):
                stack.pop()
            if not stack:
                raise ConfigError(f"Line {line_num}: list item outside of list")
            parent_list = stack[-1][1]
            parent_list.append(_parse_value(item))
            continue

        # Handle key: value
        if ':' not in key_value:
            raise ConfigError(f"Line {line_num}: expected 'key: value', got '{stripped}'")
        key, value = key_value.split(':', 1)
        key = key.strip()
        value = value.strip()

        # Pop stack to correct indentation level
        while stack and indent <= stack[-1][0]:
            stack.pop()

        if not stack:
            raise ConfigError(f"Line {line_num}: indentation error")

        parent = stack[-1][1]
        if not isinstance(parent, dict):
            raise ConfigError(f"Line {line_num}: cannot assign to non-dict")

        # Parse the value
        if value == '':
            # Empty value - could be nested dict or list
            parsed = {}
            parent[key] = parsed
            stack.append((indent, parsed))
        else:
            parsed = _parse_value(value)
            parent[key] = parsed

    return result


def _parse_value(value: str) -> Any:
    """Parse a YAML value string into Python type."""
    value = value.strip()
    if not value or value.lower() == 'null' or value == '~':
        return None
    if value.lower() == 'true':
        return True
    if value.lower() == 'false':
        return False
    
    # Handle empty list
    if value == '[]':
        return []
    
    # Try number
    try:
        if '.' in value or 'e' in value.lower():
            return float(value)
        return int(value)
    except ValueError:
        pass

    # String (strip quotes if present)
    if (value.startswith('"') and value.endswith('"')) or \
       (value.startswith("'") and value.endswith("'")):
        return value[1:-1]

    return value


class Config:
    """Configuration container with validation and defaults."""

    DEFAULTS = {
        "model": {
            "vocab_size": 1000,
            "ctx": 50,
            "emb": 50,
            "hidden": 50,
        },
        "training": {
            "steps": 100,
            "lr": 0.1,              # Higher LR for faster initial convergence
            "batch_size": 32,       # Larger batch for better gradient estimates
            "n_negatives": 256,
            "workers": None,
            "lr_scheduler": "cosine_warmup",
            "warmup_steps": 50,     # Shorter warmup
            "min_lr": 1e-5,         # Non-zero min LR
            "grad_clip": 1.0,
            "val_split": 0.1,
            "eval_every": 50,
            "early_stopping": True,
            "patience": 10,         # More patience
            "min_delta": 1e-4,
            "checkpoint_every": 100,
            "checkpoint_dir": "checkpoints",
            "auto_adjust": True,
            "auto_factor": 0.7,     # Less aggressive LR reduction
            "auto_patience": 5,     # Faster response to plateaus
            "auto_clip_relax": True,
            "momentum": 0.95,       # Higher momentum
            "rollback_window": 2,
            "optimizer": "sgd",       # sgd/apollo/apollo_mini/adam/lion/muon (UI uses train_parallel: sgd/adam/lion/muon)
            "batch_steps": 1,        # Batches computed per update (gradient accumulation; >1 loads GPU more)
            "use_gpu": True,         # Route worker grads through CUDA when available (auto-fallback to C/Python)
            # New features - enabled by default for faster convergence
            "use_frequency_neg_sampling": True,   # Better negative sampling
            "n_negatives_start": 512,             # Start with more negatives
            "n_negatives_end": 64,
            "n_negatives_decay": "cosine",        # Smoother decay
            "use_cosine_restarts": True,          # SGDR-style restarts
            "restart_period": 0,                  # Auto (steps // 10)
            "cyclic_lr": False,
            "cycle_length": 0,
        },
        "generation": {
            "max_tokens": 0,  # 0 = unlimited, structural markers end generation
            "min_tokens": 0,
            "temperature": 0.8,
            "top_k": 0,
            "top_p": 0.0,
            "repetition_penalty": 1.0,
            "stop_token_ids": [],
            "stop_strings": [],
        },
        "tokenizer": {
            "vocab_size": 1000,
            "min_frequency": 2,
        },
        "ui": {
            "theme": "dark",  # dark, light, system
        },
    }

    SCHEMA = {
        "model": {
            "vocab_size": (int, lambda x: x >= 256, "vocab_size must be >= 256"),
            "ctx": (int, lambda x: x > 0, "ctx must be > 0"),
            "emb": (int, lambda x: x > 0, "emb must be > 0"),
            "hidden": (int, lambda x: x > 0, "hidden must be > 0"),
        },
        "training": {
            "steps": (int, lambda x: x > 0, "steps must be > 0"),
            "lr": (float, lambda x: x > 0, "lr must be > 0"),
            "batch_size": (int, lambda x: x > 0, "batch_size must be > 0"),
            "n_negatives": (int, lambda x: x >= 0, "n_negatives must be >= 0"),
            "workers": (type(None), lambda x: x is None or x > 0, "workers must be None or > 0"),
            "lr_scheduler": (str, lambda x: x in ("linear", "cosine_warmup", "constant"), "lr_scheduler must be linear, cosine_warmup, or constant"),
            "warmup_steps": (int, lambda x: x >= 0, "warmup_steps must be >= 0"),
            "min_lr": (float, lambda x: x >= 0, "min_lr must be >= 0"),
            "grad_clip": (float, lambda x: x >= 0, "grad_clip must be >= 0"),
            "val_split": (float, lambda x: 0.0 <= x < 1.0, "val_split must be in [0.0, 1.0)"),
            "eval_every": (int, lambda x: x >= 0, "eval_every must be >= 0"),
            "early_stopping": (bool, lambda x: True, "early_stopping must be boolean"),
            "patience": (int, lambda x: x > 0, "patience must be > 0"),
            "min_delta": (float, lambda x: x >= 0, "min_delta must be >= 0"),
            "checkpoint_every": (int, lambda x: x >= 0, "checkpoint_every must be >= 0"),
            "checkpoint_dir": (str, lambda x: True, "checkpoint_dir must be string"),
            "auto_adjust": (bool, lambda x: True, "auto_adjust must be boolean"),
            "auto_factor": (float, lambda x: 0.0 < x < 1.0, "auto_factor must be in (0.0, 1.0)"),
            "auto_patience": (int, lambda x: x > 0, "auto_patience must be > 0"),
            "auto_clip_relax": (bool, lambda x: True, "auto_clip_relax must be boolean"),
            "momentum": (float, lambda x: 0.0 <= x < 1.0, "momentum must be in [0.0, 1.0)"),
            "rollback_window": (int, lambda x: x >= 0, "rollback_window must be >= 0"),
            "optimizer": (str, lambda x: x in ("sgd", "apollo", "apollo_mini", "adam", "lion", "muon"),
                          "optimizer must be sgd, apollo, apollo_mini, adam, lion, or muon"),
            "batch_steps": (int, lambda x: x >= 1, "batch_steps must be >= 1"),
            "use_gpu": (bool, lambda x: True, "use_gpu must be boolean"),
            # New features
            "use_frequency_neg_sampling": (bool, lambda x: True, "use_frequency_neg_sampling must be boolean"),
            "n_negatives_start": (type(None), lambda x: x is None or (isinstance(x, int) and x > 0), "n_negatives_start must be None or positive int"),
            "n_negatives_end": (int, lambda x: x >= 0, "n_negatives_end must be >= 0"),
            "n_negatives_decay": (str, lambda x: x in ("exponential", "linear", "cosine"), "n_negatives_decay must be exponential, linear, or cosine"),
            "use_cosine_restarts": (bool, lambda x: True, "use_cosine_restarts must be boolean"),
            "restart_period": (int, lambda x: x >= 0, "restart_period must be >= 0"),
            "cyclic_lr": (bool, lambda x: True, "cyclic_lr must be boolean"),
            "cycle_length": (int, lambda x: x >= 0, "cycle_length must be >= 0"),
        },
        "generation": {
            "max_tokens": (int, lambda x: x >= 0, "max_tokens must be >= 0 (0 = unlimited)"),
            "min_tokens": (int, lambda x: x >= 0, "min_tokens must be >= 0"),
            "temperature": (float, lambda x: x > 0, "temperature must be > 0"),
            "top_k": (int, lambda x: x >= 0, "top_k must be >= 0"),
            "top_p": (float, lambda x: 0.0 <= x <= 1.0, "top_p must be between 0.0 and 1.0"),
            "repetition_penalty": (float, lambda x: x >= 1.0, "repetition_penalty must be >= 1.0"),
            "stop_token_ids": (list, lambda x: all(isinstance(i, int) for i in x), "stop_token_ids must be list of ints"),
            "stop_strings": (list, lambda x: all(isinstance(s, str) for s in x), "stop_strings must be list of strings"),
        },
        "tokenizer": {
            "vocab_size": (int, lambda x: x >= 256, "vocab_size must be >= 256"),
            "min_frequency": (int, lambda x: x > 0, "min_frequency must be > 0"),
        },
        "ui": {
            "theme": (str, lambda x: x in ("dark", "light", "system"), "ui.theme must be dark, light, or system"),
        },
    }

    def __init__(self, data: Optional[Dict[str, Any]] = None):
        self._data = self._merge_defaults(data or {})
        self._validate()

    def _merge_defaults(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge user config with defaults."""
        result = {}
        for key, default_val in self.DEFAULTS.items():
            if key in data:
                user_val = data[key]
                if isinstance(default_val, dict) and isinstance(user_val, dict):
                    result[key] = self._merge_dicts(default_val, user_val)
                else:
                    result[key] = user_val
            else:
                result[key] = default_val
        return result

    def _merge_dicts(self, default: Dict, user: Dict) -> Dict:
        result = default.copy()
        for k, v in user.items():
            if k in result and isinstance(result[k], dict) and isinstance(v, dict):
                result[k] = self._merge_dicts(result[k], v)
            else:
                result[k] = v
        return result

    def _validate(self):
        """Validate config against schema."""
        for section, rules in self.SCHEMA.items():
            if section not in self._data:
                raise ConfigError(f"Missing required section: {section}")
            section_data = self._data[section]
            for key, (expected_type, validator, msg) in rules.items():
                if key not in section_data:
                    raise ConfigError(f"Missing required key: {section}.{key}")
                val = section_data[key]
                if expected_type is not type(None) and not isinstance(val, expected_type):
                    raise ConfigError(
                        f"{section}.{key}: expected {expected_type.__name__}, got {type(val).__name__}"
                    )
                if not validator(val):
                    raise ConfigError(f"{section}.{key}: {msg}")

    @classmethod
    def from_file(cls, path: Union[str, Path]) -> "Config":
        """Load config from YAML or JSON file."""
        path = Path(path)
        text = path.read_text(encoding="utf-8")
        if path.suffix in (".yaml", ".yml"):
            data = load_yaml(text)
        elif path.suffix == ".json":
            data = json.loads(text)
        else:
            raise ConfigError(f"Unsupported config format: {path.suffix}")
        return cls(data)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Config":
        """Create config from dictionary."""
        return cls(data)

    def to_dict(self) -> Dict[str, Any]:
        """Return config as dictionary."""
        return self._data.copy()

    def get(self, section: str, key: str, default: Any = None) -> Any:
        """Get a config value with optional default."""
        return self._data.get(section, {}).get(key, default)

    def __getitem__(self, section: str) -> Dict[str, Any]:
        return self._data[section]

    def __contains__(self, section: str) -> bool:
        return section in self._data


# Convenience functions for common configs
def model_config(vocab_size: int = 1000, ctx: int = 50, emb: int = 50, hidden: int = 50) -> Dict[str, Any]:
    """Create a model config dict."""
    return {"model": {"vocab_size": vocab_size, "ctx": ctx, "emb": emb, "hidden": hidden}}


def training_config(steps: int = 100, lr: float = 0.05, batch_size: int = 16,
                    n_negatives: int = 256, workers: Optional[int] = None,
                    lr_scheduler: str = "cosine_warmup", warmup_steps: int = 100,
                    min_lr: float = 0.0, grad_clip: float = 1.0,
                    val_split: float = 0.1, eval_every: int = 50,
                    early_stopping: bool = True, patience: int = 5,
                    min_delta: float = 1e-4, checkpoint_every: int = 100,
                    checkpoint_dir: str = "checkpoints",
                    auto_adjust: bool = True, auto_factor: float = 0.5,
                    auto_patience: int = 10, auto_clip_relax: bool = True,
                    momentum: float = 0.9, rollback_window: int = 2,
                    # New features
                    use_frequency_neg_sampling: bool = False,
                    n_negatives_start: Optional[int] = None,
                    n_negatives_end: int = 64,
                    n_negatives_decay: str = "exponential",
                    use_cosine_restarts: bool = False,
                    restart_period: int = 0,
                    cyclic_lr: bool = False,
                    cycle_length: int = 0) -> Dict[str, Any]:
    """Create a training config dict."""
    return {"training": {"steps": steps, "lr": lr, "batch_size": batch_size,
                         "n_negatives": n_negatives, "workers": workers,
                         "lr_scheduler": lr_scheduler, "warmup_steps": warmup_steps,
                         "min_lr": min_lr, "grad_clip": grad_clip,
                         "val_split": val_split, "eval_every": eval_every,
                         "early_stopping": early_stopping, "patience": patience,
                         "min_delta": min_delta, "checkpoint_every": checkpoint_every,
                         "checkpoint_dir": checkpoint_dir,
                         "auto_adjust": auto_adjust, "auto_factor": auto_factor,
                         "auto_patience": auto_patience, "auto_clip_relax": auto_clip_relax,
                         "momentum": momentum, "rollback_window": rollback_window,
                         "use_frequency_neg_sampling": use_frequency_neg_sampling,
                         "n_negatives_start": n_negatives_start,
                         "n_negatives_end": n_negatives_end,
                         "n_negatives_decay": n_negatives_decay,
                         "use_cosine_restarts": use_cosine_restarts,
                         "restart_period": restart_period,
                         "cyclic_lr": cyclic_lr,
                         "cycle_length": cycle_length}}


def generation_config(max_tokens: int = 0, min_tokens: int = 0, temperature: float = 0.8, 
                      top_k: int = 0, top_p: float = 0.0, repetition_penalty: float = 1.0,
                      stop_token_ids: Optional[List[int]] = None,
                      stop_strings: Optional[List[str]] = None) -> Dict[str, Any]:
    """Create a generation config dict."""
    return {"generation": {"max_tokens": max_tokens, "min_tokens": min_tokens, 
                           "temperature": temperature, "top_k": top_k, "top_p": top_p, 
                           "repetition_penalty": repetition_penalty,
                           "stop_token_ids": stop_token_ids or [],
                           "stop_strings": stop_strings or []}}


def tokenizer_config(vocab_size: int = 1000, min_frequency: int = 2) -> Dict[str, Any]:
    """Create a tokenizer config dict."""
    return {"tokenizer": {"vocab_size": vocab_size, "min_frequency": min_frequency}}


def merge_configs(*configs: Dict[str, Any]) -> Dict[str, Any]:
    """Merge multiple config dicts (later overrides earlier)."""
    result = {}
    for cfg in configs:
        for section, values in cfg.items():
            if section in result and isinstance(result[section], dict) and isinstance(values, dict):
                result[section] = merge_dicts(result[section], values)
            else:
                result[section] = values
    return result


def merge_dicts(a: Dict, b: Dict) -> Dict:
    """Deep merge two dicts (b overrides a)."""
    result = a.copy()
    for k, v in b.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = merge_dicts(result[k], v)
        else:
            result[k] = v
    return result


if __name__ == "__main__":
    # Demo
    import tempfile

    yaml_text = """
model:
  vocab_size: 5000
  ctx: 100
  emb: 64
  hidden: 128
training:
  steps: 500
  lr: 0.01
  batch_size: 32
generation:
  n_tokens: 100
  temperature: 0.7
"""

    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(yaml_text)
        yaml_path = f.name

    json_path = yaml_path.replace('.yaml', '.json')
    with open(json_path, 'w') as f:
        json.dump({"model": {"vocab_size": 3000}}, f)

    try:
        cfg1 = Config.from_file(yaml_path)
        print("YAML config:", cfg1.to_dict())

        cfg2 = Config.from_file(json_path)
        print("JSON config:", cfg2.to_dict())

        cfg3 = Config.from_dict({})
        print("Defaults:", cfg3.to_dict())

        # Test validation
        try:
            Config.from_dict({"model": {"vocab_size": 100}})  # Too small
        except ConfigError as e:
            print("Validation caught:", e)

    finally:
        os.unlink(yaml_path)
        os.unlink(json_path)