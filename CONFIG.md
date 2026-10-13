# Configuration Reference (CONFIG.md)

All configuration options for ScratchpadLM / TinyLM. Configuration is loaded from `config.yaml` (preferred) or `config.json` at startup, with sensible defaults for any missing values.

## Configuration Sections

### `model` — Base Model Architecture

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `vocab_size` | int | `1000` | Vocabulary size (must be ≥ 256). Updated automatically when tokenizer extends. |
| `ctx` | int | `50` | Context window size (number of previous tokens). |
| `emb` | int | `50` | Token embedding dimension. |
| `hidden` | int | `50` | Hidden layer dimension (TinyLM) or transformer `emb` (Router/Specialized). |

**Validation**: All values must be positive integers. `vocab_size ≥ 256`.

---

### `training` — Training Hyperparameters

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `steps` | int | `100` | Number of training steps. |
| `lr` | float | `0.05` | Base learning rate. |
| `batch_size` | int | `16` | Number of context windows per step. |
| `n_negatives` | int | `256` | Number of negative samples for sampled softmax. |
| `workers` | int \| null | `null` | Parallel workers (`null` = CPU count). |
| `lr_scheduler` | string | `"cosine_warmup"` | One of: `"cosine_warmup"`, `"linear"`, `"constant"`. |
| `warmup_steps` | int | `100` | Warmup steps for `cosine_warmup`/`linear` schedulers. |
| `min_lr` | float | `0.0` | Minimum learning rate after decay. |
| `grad_clip` | float | `1.0` | Global gradient norm clipping (0 = disabled). |
| `val_split` | float | `0.1` | Fraction of data held out for validation (0.0–1.0). |
| `eval_every` | int | `50` | Evaluate validation loss every N steps (0 = disabled). |
| `early_stopping` | bool | `true` | Enable early stopping on validation loss. |
| `patience` | int | `5` | Steps without improvement before stopping. |
| `min_delta` | float | `1e-4` | Minimum improvement to reset patience. |
| `checkpoint_every` | int | `100` | Save checkpoint every N steps (0 = disabled). |
| `checkpoint_dir` | string | `"checkpoints"` | Directory for checkpoint files. |
| `auto_adjust` | bool | `true` | Enable automatic LR reduction + clip relaxation. |
| `auto_factor` | float | `0.5` | LR multiplier on plateau (0.0–1.0). |
| `auto_patience` | int | `10` | Steps of plateau before LR reduction. |
| `auto_clip_relax` | bool | `true` | Relax grad clip limit if chronic clipping detected. |
| `momentum` | float | `0.9` | SGD momentum (0 = plain SGD). |
| `rollback_window` | int | `2` | Snapshot every step; restore lowest-loss step in window on spike (0 = disabled). |

**Validation**:
- `steps`, `batch_size`, `n_negatives` ≥ 0 (`batch_size`, `n_negatives` > 0 in practice)
- `lr`, `min_lr`, `grad_clip` ≥ 0
- `val_split` ∈ [0.0, 1.0)
- `lr_scheduler` ∈ {"cosine_warmup", "linear", "constant"}
- `auto_factor` ∈ (0.0, 1.0)
- Boolean fields must be `true`/`false`

---

### `generation` — Text Generation Parameters

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `n_tokens` | int | `60` | Number of tokens to generate. |
| `temperature` | float | `0.8` | Sampling temperature (> 0). |
| `top_k` | int | `0` | Top-k sampling (0 = disabled). |
| `top_p` | float | `0.0` | Nucleus/top-p sampling (0.0 = disabled, 1.0 = all). |
| `repetition_penalty` | float | `1.0` | Penalty for repeated tokens (≥ 1.0). |

**Validation**:
- `n_tokens` > 0
- `temperature` > 0
- `top_k` ≥ 0
- `top_p` ∈ [0.0, 1.0]
- `repetition_penalty` ≥ 1.0

---

### `tokenizer` — Tokenizer Settings

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `vocab_size` | int | `1000` | Target vocabulary size for `extend()` (≥ 256). |
| `min_frequency` | int | `2` | Minimum pair frequency for new merges. |

**Validation**: Both must be positive integers, `vocab_size ≥ 256`.

---

### `ui` — User Interface Settings

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `theme` | string | `"system"` | UI theme: `"dark"`, `"light"`, or `"system"` (follows OS). |

**Validation**: Must be one of `"dark"`, `"light"`, `"system"`.

---

## Complete Example (config.yaml)

```yaml
model:
  vocab_size: 2000
  ctx: 128
  emb: 64
  hidden: 128

training:
  steps: 2000
  lr: 0.05
  batch_size: 32
  n_negatives: 512
  workers: 4
  lr_scheduler: "cosine_warmup"
  warmup_steps: 200
  min_lr: 0.001
  grad_clip: 1.0
  val_split: 0.15
  eval_every: 100
  early_stopping: true
  patience: 10
  min_delta: 1e-4
  checkpoint_every: 200
  checkpoint_dir: "checkpoints"
  auto_adjust: true
  auto_factor: 0.5
  auto_patience: 10
  auto_clip_relax: true
  momentum: 0.9
  rollback_window: 2

generation:
  n_tokens: 100
  temperature: 0.7
  top_k: 50
  top_p: 0.9
  repetition_penalty: 1.1

tokenizer:
  vocab_size: 2000
  min_frequency: 2

ui:
  theme: "system"
```

---

## Loading Configuration

### Python API

```python
from config import Config

# Load from file
cfg = Config.from_file("config.yaml")

# Or from dict
cfg = Config.from_dict({"model": {"vocab_size": 2000}})

# Access values
vocab = cfg["model"]["vocab_size"]
lr = cfg.get("training", "lr", 0.05)  # with default

# Save modified config
cfg._data["training"]["steps"] = 500
from ui_base import write_config_yaml
write_config_yaml("config.yaml", cfg.to_dict())
```

### CLI

```bash
# Config is loaded automatically by CLI commands
python3 -m main --cli train --steps 500  # overrides config temporarily
```

---

## Validation Behavior

- Missing sections/keys are filled from `DEFAULTS`
- Type mismatches raise `ConfigError`
- Custom validators (range, enum) raise `ConfigError` with descriptive message
- Validation happens at `Config` construction time

---

## Environment Overrides

| Env Var | Effect |
|---------|--------|
| `SCRATCHPADLM_UI` | Force UI toolkit: `pyside6` or `tkinter` |

---

## Programmatic Defaults

```python
from config import Config, model_config, training_config, generation_config, tokenizer_config, merge_configs

# Build custom config dicts
model_cfg = model_config(vocab_size=5000, ctx=256, emb=128, hidden=256)
train_cfg = training_config(steps=5000, lr=0.01, batch_size=64)
gen_cfg = generation_config(n_tokens=200, temperature=0.7)
tok_cfg = tokenizer_config(vocab_size=5000)

# Merge (later overrides earlier)
full = merge_configs(model_cfg, train_cfg, gen_cfg, tok_cfg)
cfg = Config.from_dict(full)
```

---

## Notes

- `vocab_size` in `model` section should match tokenizer's actual vocab size after extension. The UI and training pipeline handle this automatically via `sync_vocab()`.
- `ctx` affects memory usage quadratically in attention-based models (Router, SpecializedLM). Keep ≤ 256 for ≤100K param models.
- `n_negatives` trades off training speed vs. gradient quality. 256 is a good default; 512 for higher quality.
- `auto_adjust` combines plateau LR reduction (like ReduceLROnPlateau) with gradient clip relaxation when clipping fires consecutively.
- Theme `"system"` uses Qt's `QStyleHints.colorScheme` on Qt, falls back to `gsettings`/`XDG_CURRENT_DESKTOP` on Linux.