"""Model and Task Registries. Stdlib only."""
from typing import Any, Callable, Dict, List, Optional
from dataclasses import dataclass, field
import json
from pathlib import Path


# =============================================================================
# Model Registry
# =============================================================================

ModelFactory = Callable[..., Any]  # Function that returns a model instance


class ModelRegistry:
    """Registry for model factories. Allows registering models by name."""

    def __init__(self):
        self._factories: Dict[str, ModelFactory] = {}
        self._metadata: Dict[str, Dict[str, Any]] = {}

    def register(self, name: str, factory: ModelFactory, **metadata) -> None:
        """Register a model factory with optional metadata.

        Args:
            name: Unique model name (e.g., "tiny-code", "router")
            factory: Callable that returns a model instance
            **metadata: Arbitrary metadata (description, param_count, domain, etc.)
        """
        if name in self._factories:
            raise ValueError(f"Model '{name}' already registered")
        self._factories[name] = factory
        self._metadata[name] = metadata

    def get(self, name: str) -> ModelFactory:
        """Get a model factory by name."""
        if name not in self._factories:
            raise KeyError(f"Model '{name}' not found. Available: {list(self._factories)}")
        return self._factories[name]

    def create(self, name: str, *args, **kwargs) -> Any:
        """Create a model instance by name."""
        return self.get(name)(*args, **kwargs)

    def list(self) -> List[str]:
        """List all registered model names."""
        return list(self._factories.keys())

    def metadata(self, name: str) -> Dict[str, Any]:
        """Get metadata for a model."""
        return self._metadata.get(name, {})

    def has(self, name: str) -> bool:
        """Check if model is registered."""
        return name in self._factories

    def remove(self, name: str) -> bool:
        """Remove a model from registry. Returns True if removed."""
        if name in self._factories:
            del self._factories[name]
            del self._metadata[name]
            return True
        return False

    def clear(self) -> None:
        """Clear all registered models."""
        self._factories.clear()
        self._metadata.clear()


# Global model registry instance
MODEL_REGISTRY = ModelRegistry()


# =============================================================================
# Task Registry
# =============================================================================

@dataclass
class TaskTemplate:
    """Template for a task that can be instantiated with a model."""
    name: str
    model_name: str
    prompt_template: str
    output_parser: Optional[Callable[[str], Any]] = None
    description: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)


class TaskRegistry:
    """Registry for task templates."""

    def __init__(self):
        self._tasks: Dict[str, TaskTemplate] = {}

    def register(self, task: TaskTemplate) -> None:
        """Register a task template."""
        if task.name in self._tasks:
            raise ValueError(f"Task '{task.name}' already registered")
        self._tasks[task.name] = task

    def get(self, name: str) -> TaskTemplate:
        """Get a task template by name."""
        if name not in self._tasks:
            raise KeyError(f"Task '{name}' not found. Available: {list(self._tasks)}")
        return self._tasks[name]

    def list(self) -> List[str]:
        """List all registered task names."""
        return list(self._tasks.keys())

    def has(self, name: str) -> bool:
        """Check if task is registered."""
        return name in self._tasks

    def remove(self, name: str) -> bool:
        """Remove a task from registry."""
        if name in self._tasks:
            del self._tasks[name]
            return True
        return False


# Global task registry instance
TASK_REGISTRY = TaskRegistry()


# =============================================================================
# Tokenizer Manager (Singleton)
# =============================================================================

class TokenizerManager:
    """Singleton manager for the shared BPETokenizer instance."""

    _instance: Optional['TokenizerManager'] = None
    _tokenizer: Optional[Any] = None  # BPETokenizer
    _vocab_size: int = 1000
    _min_frequency: int = 2

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._tokenizer is None:
            from tokenizer import BPETokenizer
            self._tokenizer = BPETokenizer()

    @property
    def tokenizer(self):
        """Get the shared tokenizer instance."""
        return self._tokenizer

    def load(self, path: str) -> None:
        """Load tokenizer from file."""
        self._tokenizer.load(path)

    def save(self, path: str) -> None:
        """Save tokenizer to file."""
        self._tokenizer.save(path)

    def extend(self, text: str, vocab_size: Optional[int] = None, min_frequency: Optional[int] = None) -> None:
        """Extend tokenizer vocabulary with new text."""
        vs = vocab_size or self._vocab_size
        mf = min_frequency or self._min_frequency
        self._tokenizer.extend(text, vocab_size=vs, min_frequency=mf)
        self._vocab_size = max(self._vocab_size, vs)

    def train(self, text: str, vocab_size: int, min_frequency: int = 2, workers: Optional[int] = None) -> None:
        """Train tokenizer from scratch."""
        self._tokenizer.train(text, vocab_size, min_frequency, workers)
        self._vocab_size = vocab_size
        self._min_frequency = min_frequency

    @property
    def vocab_size(self) -> int:
        return len(self._tokenizer.vocab)

    @property
    def vocab(self) -> Dict[int, bytes]:
        return self._tokenizer.vocab

    @property
    def merges(self) -> Dict[tuple, int]:
        return self._tokenizer.merges

    def encode(self, text: str) -> List[int]:
        return self._tokenizer.encode(text)

    def decode(self, ids: List[int]) -> str:
        return self._tokenizer.decode(ids)

    def reset(self) -> None:
        """Reset to fresh tokenizer (for testing)."""
        from tokenizer import BPETokenizer
        self._tokenizer = BPETokenizer()
        self._vocab_size = 1000
        self._min_frequency = 2


# Global tokenizer manager instance
TOKENIZER_MANAGER = TokenizerManager()


# =============================================================================
# Convenience Functions
# =============================================================================

def register_model(name: str, factory: ModelFactory, **metadata) -> None:
    """Register a model in the global registry."""
    MODEL_REGISTRY.register(name, factory, **metadata)


def get_model_factory(name: str) -> ModelFactory:
    """Get a model factory from the global registry."""
    return MODEL_REGISTRY.get(name)


def create_model(name: str, *args, **kwargs) -> Any:
    """Create a model instance from the global registry."""
    return MODEL_REGISTRY.create(name, *args, **kwargs)


def list_models() -> List[str]:
    """List all registered models."""
    return MODEL_REGISTRY.list()


def register_task(task: TaskTemplate) -> None:
    """Register a task in the global registry."""
    TASK_REGISTRY.register(task)


def get_task(name: str) -> TaskTemplate:
    """Get a task from the global registry."""
    return TASK_REGISTRY.get(name)


def list_tasks() -> List[str]:
    """List all registered tasks."""
    return TASK_REGISTRY.list()


def get_tokenizer() -> Any:
    """Get the shared tokenizer instance."""
    return TOKENIZER_MANAGER.tokenizer


# Register built-in model factories (run on import)
from ai import TinyLM, Router, RouterConfig, SpecializedLM, SpecializedConfig

register_model("TinyLM", lambda **kw: TinyLM(**kw),
               description="TinyLM base architecture", domain="general")
register_model("Router", lambda **kw: Router(RouterConfig(**kw)),
               description="Router model (2-layer transformer)", domain="routing", param_count=300000)
register_model("SpecializedLM", lambda **kw: SpecializedLM(SpecializedConfig(**kw)),
               description="Specialized model (1-layer transformer, ≤100K params)", domain="specialized", param_count=100000)


if __name__ == "__main__":
    # Demo
    from ai import TinyLM

    # Register some models
    register_model("tiny-base", lambda: TinyLM(vocab_size=1000, ctx=50, emb=50, hidden=50),
                   description="Base model", param_count=50000, domain="general")
    register_model("tiny-code", lambda: TinyLM(vocab_size=1000, ctx=50, emb=64, hidden=128),
                   description="Code-specialized", param_count=80000, domain="code")

    print("Registered models:", list_models())
    print("tiny-code metadata:", MODEL_REGISTRY.metadata("tiny-code"))

    # Create models
    model1 = create_model("tiny-base")
    model2 = create_model("tiny-code")
    print(f"Created: {type(model1).__name__}, {type(model2).__name__}")

    # Register tasks
    register_task(TaskTemplate(
        name="generate",
        model_name="tiny-base",
        prompt_template="Continue: {prompt}",
        description="Text generation"
    ))
    register_task(TaskTemplate(
        name="code",
        model_name="tiny-code",
        prompt_template="Write code: {prompt}",
        description="Code generation"
    ))

    print("Registered tasks:", list_tasks())
    task = get_task("code")
    print(f"Task: {task.name}, model: {task.model_name}")

    # Test tokenizer manager
    tok = get_tokenizer()
    tok.train("hello world " * 100, vocab_size=500)
    print(f"Tokenizer vocab size: {TOKENIZER_MANAGER.vocab_size}")
    print(f"Encode 'hello': {TOKENIZER_MANAGER.encode('hello')}")