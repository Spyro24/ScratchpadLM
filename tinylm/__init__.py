"""ScratchpadLM core package (stdlib only).

Split of the former monolithic `ai.py` into focused modules.
`ai.py` remains as a backward-compatible shim re-exporting this API.
"""
from tinylm.autograd import Value, dot, matvec_dot, softmax_cross_entropy
from tinylm.serialization import (
    MAGIC,
    sync_vocab,
    load_corpus,
    save_model,
    load_model,
    count_params,
    enforce_param_limit,
    _quantize_rows,
    _dequantize_rows,
)
from tinylm.models_tiny import TinyLM
from tinylm.transformer import (
    layer_norm,
    fast_layer_norm,
    attention,
    fast_attention,
    feed_forward,
    fast_feed_forward,
    TransformerBlock,
)
from tinylm.router import RouterConfig, Router
from tinylm.runtime import ModelRuntime
from tinylm.specialized import (
    SpecializedConfig,
    SpecializedLM,
    create_specialized_model,
    get_specialized_presets,
)
from tinylm.router_data import (
    RouterTrainingExample,
    encode_router_example,
    decode_router_sequence,
    create_router_training_data,
    train_router,
    generate_router_training_data,
    _MODEL_SLOTS,
    _stable_hash,
    _model_slot,
    _ensure_vocab_size,
)
from tinylm.data_clean import (
    BOILERPLATE_PATTERNS,
    DOMAIN_KEYWORDS_CLEAN,
    normalize_text,
    strip_boilerplate,
    deduplicate_lines,
    classify_by_domain,
    sort_and_classify_data,
    clean_training_data,
)
from tinylm.sched import (
    LR_Scheduler,
    clip_grads,
    AutoAdjust,
    evaluate,
    EarlyStopping,
)
from tinylm.checkpoint import (
    Checkpoint,
    save_checkpoint,
    load_checkpoint,
    resume_from_checkpoint,
    _flat_params,
    _set_flat_params,
)
from tinylm.training import (
    train,
    _train_c,
    _accel_available,
    _flat_data,
)
from tinylm.parallel import (
    train_parallel,
    _WorkerState,
    _init_worker,
    _worker_grads_v2,
)
from tinylm.generation import (
    fast_logits,
    _sample_next_token,
    generate,
    generate_with_preview,
)

__all__ = [
    "Value", "dot", "matvec_dot", "softmax_cross_entropy",
    "MAGIC", "sync_vocab", "load_corpus", "save_model", "load_model",
    "count_params", "enforce_param_limit",
    "_quantize_rows", "_dequantize_rows",
    "_MODEL_SLOTS", "_stable_hash", "_model_slot", "_ensure_vocab_size",
    "BOILERPLATE_PATTERNS", "DOMAIN_KEYWORDS_CLEAN",
    "TinyLM",
    "layer_norm", "fast_layer_norm", "attention", "fast_attention",
    "feed_forward", "fast_feed_forward", "TransformerBlock",
    "RouterConfig", "Router",
    "ModelRuntime",
    "SpecializedConfig", "SpecializedLM", "create_specialized_model",
    "get_specialized_presets",
    "RouterTrainingExample", "encode_router_example", "decode_router_sequence",
    "create_router_training_data", "train_router", "generate_router_training_data",
    "normalize_text", "strip_boilerplate", "deduplicate_lines",
    "classify_by_domain", "sort_and_classify_data", "clean_training_data",
    "LR_Scheduler", "clip_grads", "AutoAdjust", "evaluate", "EarlyStopping",
    "Checkpoint", "save_checkpoint", "load_checkpoint", "resume_from_checkpoint",
    "_flat_params", "_set_flat_params",
    "train", "_train_c", "_accel_available", "_flat_data",
    "train_parallel", "_WorkerState", "_init_worker", "_worker_grads_v2",
    "fast_logits", "_sample_next_token", "generate", "generate_with_preview",
]
