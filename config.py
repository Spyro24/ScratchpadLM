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
            "lr": 0.05,
            "batch_size": 16,
            "n_negatives": 256,
            "workers": None,
            "lr_scheduler": "cosine_warmup",
            "warmup_steps": 100,
            "min_lr": 0.0,
            "grad_clip": 1.0,
            "val_split": 0.1,
            "eval_every": 50,
            "early_stopping": True,
            "patience": 5,
            "min_delta": 1e-4,
            "checkpoint_every": 100,
            "checkpoint_dir": "checkpoints",
            "auto_adjust": True,
            "auto_factor": 0.5,
            "auto_patience": 10,
            "auto_clip_relax": True,
        },
        "generation": {
            "n_tokens": 60,
            "temperature": 0.8,
            "top_k": 0,
            "top_p": 0.0,
            "repetition_penalty": 1.0,
        },
        "tokenizer": {
            "vocab_size": 1000,
            "min_frequency": 2,
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
        },
        "generation": {
            "n_tokens": (int, lambda x: x > 0, "n_tokens must be > 0"),
            "temperature": (float, lambda x: x > 0, "temperature must be > 0"),
            "top_k": (int, lambda x: x >= 0, "top_k must be >= 0"),
            "top_p": (float, lambda x: 0.0 <= x <= 1.0, "top_p must be between 0.0 and 1.0"),
            "repetition_penalty": (float, lambda x: x >= 1.0, "repetition_penalty must be >= 1.0"),
        },
        "tokenizer": {
            "vocab_size": (int, lambda x: x >= 256, "vocab_size must be >= 256"),
            "min_frequency": (int, lambda x: x > 0, "min_frequency must be > 0"),
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
                    auto_patience: int = 10, auto_clip_relax: bool = True) -> Dict[str, Any]:
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
                         "auto_patience": auto_patience, "auto_clip_relax": auto_clip_relax}}


def generation_config(n_tokens: int = 60, temperature: float = 0.8, top_k: int = 0, top_p: float = 0.0, repetition_penalty: float = 1.0) -> Dict[str, Any]:
    """Create a generation config dict."""
    return {"generation": {"n_tokens": n_tokens, "temperature": temperature, "top_k": top_k, "top_p": top_p, "repetition_penalty": repetition_penalty}}


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