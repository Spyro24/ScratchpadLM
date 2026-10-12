"""Dynamic Model Management: ModelSpec, ModelCatalog, ModelLoader, ModelCreator, ModelUpdater."""
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
from pathlib import Path
import json
import time
import shutil


# =============================================================================
# ModelSpec: Metadata for a specialized model
# =============================================================================

@dataclass
class ModelSpec:
    """Specification for a registered model.

    Attributes:
        name: Unique model name (e.g., "tiny-code")
        path: Path to model file (.tlm)
        domain_keywords: List of keywords for routing (e.g., ["code", "python"])
        param_count: Total parameter count
        config: Model configuration dict (from get_config())
        created_at: Unix timestamp
        metrics: Optional dict with avg_rating, train_count, last_trained, improvement_delta
    """
    name: str
    path: str
    domain_keywords: List[str]
    param_count: int
    config: Dict[str, Any]
    created_at: float = field(default_factory=time.time)
    metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelSpec":
        return cls(**data)


# =============================================================================
# ModelCatalog: Persistent registry of ModelSpecs
# =============================================================================

class ModelCatalog:
    """Persistent catalog of models stored as JSON file."""

    def __init__(self, catalog_path: str = "models/catalog.json"):
        self.catalog_path = Path(catalog_path)
        self.catalog_path.parent.mkdir(parents=True, exist_ok=True)
        self._specs: Dict[str, ModelSpec] = {}
        self._load()

    def _load(self) -> None:
        if self.catalog_path.exists():
            with open(self.catalog_path, "r") as f:
                data = json.load(f)
            for name, spec_data in data.items():
                self._specs[name] = ModelSpec.from_dict(spec_data)

    def _save(self) -> None:
        data = {name: spec.to_dict() for name, spec in self._specs.items()}
        with open(self.catalog_path, "w") as f:
            json.dump(data, f, indent=2)

    def add(self, spec: ModelSpec) -> None:
        """Add or update a model spec."""
        if spec.name in self._specs:
            raise ValueError(f"Model '{spec.name}' already exists. Use update() or remove() first.")
        self._specs[spec.name] = spec
        self._save()

    def remove(self, name: str) -> bool:
        """Remove a model spec. Returns True if removed."""
        if name in self._specs:
            del self._specs[name]
            self._save()
            return True
        return False

    def get(self, name: str) -> Optional[ModelSpec]:
        """Get a model spec by name."""
        return self._specs.get(name)

    def list(self) -> List[ModelSpec]:
        """List all model specs."""
        return list(self._specs.values())

    def find_by_domain(self, keywords: List[str]) -> List[ModelSpec]:
        """Find models matching any of the given domain keywords."""
        results = []
        for spec in self._specs.values():
            if any(kw in spec.domain_keywords for kw in keywords):
                results.append(spec)
        return results

    def update(self, name: str, **kwargs) -> bool:
        """Update fields of an existing model spec."""
        if name not in self._specs:
            return False
        spec = self._specs[name]
        for key, value in kwargs.items():
            if hasattr(spec, key):
                setattr(spec, key, value)
        self._save()
        return True

    def update_metrics(self, name: str, metrics: Dict[str, Any]) -> bool:
        """Update metrics for a model."""
        if name not in self._specs:
            return False
        self._specs[name].metrics.update(metrics)
        self._save()
        return True

    def discover_models(self, models_dir: str = "models") -> List[ModelSpec]:
        """Scan models directory for .tlm files and register any not in catalog.

        Args:
            models_dir: Directory containing model files

        Returns:
            List of newly discovered ModelSpecs
        """
        models_path = Path(models_dir)
        if not models_path.exists():
            return []

        from ai import load_model, count_params

        discovered = []
        for model_file in models_path.glob("*.tlm"):
            name = model_file.stem
            if name in self._specs:
                continue  # Already in catalog

            try:
                # Load model to get config and count params
                model = load_model(str(model_file))
                config = model.get_config()
                param_count = count_params(model)

                # Infer domain keywords from filename
                # e.g., "tiny-code" -> ["code"], "my-model" -> []
                domain_keywords = []
                if "-" in name:
                    # Use the part after the last hyphen as primary domain
                    domain_keywords.append(name.split("-")[-1])
                else:
                    domain_keywords.append(name)

                spec = ModelSpec(
                    name=name,
                    path=str(model_file),
                    domain_keywords=domain_keywords,
                    param_count=param_count,
                    config=config,
                    created_at=time.time(),
                    metrics={}
                )
                self._specs[name] = spec
                discovered.append(spec)
                print(f"Discovered: {name} ({param_count:,} params, domains: {domain_keywords})")
            except Exception as e:
                print(f"Failed to load {model_file}: {e}")

        if discovered:
            self._save()

        return discovered


# =============================================================================
# ModelLoader: Load models from catalog
# =============================================================================

class ModelLoader:
    """Loads models from catalog, injects shared tokenizer, validates param limits."""

    def __init__(self, catalog: ModelCatalog, tokenizer_manager=None, max_params: Optional[int] = None):
        self.catalog = catalog
        self.tokenizer_manager = tokenizer_manager
        self.max_params = max_params
        self._loaded_models: Dict[str, Any] = {}

    def load_model(self, name: str, force_reload: bool = False) -> Any:
        """Load a model by name from catalog.

        Args:
            name: Model name from catalog
            force_reload: If True, reload even if already in memory

        Returns:
            Model instance with shared tokenizer injected
        """
        if name in self._loaded_models and not force_reload:
            return self._loaded_models[name]

        spec = self.catalog.get(name)
        if spec is None:
            raise KeyError(f"Model '{name}' not found in catalog")

        # Load model from file
        from ai import load_model
        model = load_model(spec.path)

        # Validate param limit
        if self.max_params is not None:
            from ai import count_params
            n_params = count_params(model)
            if n_params > self.max_params:
                raise ValueError(f"Model '{name}' has {n_params:,} params, exceeds limit {self.max_params:,}")

        # Inject shared tokenizer if available
        if self.tokenizer_manager is not None:
            # Model should have a way to accept tokenizer
            if hasattr(model, 'set_tokenizer'):
                model.set_tokenizer(self.tokenizer_manager.tokenizer)
            # Or store reference
            model._tokenizer = self.tokenizer_manager.tokenizer

        self._loaded_models[name] = model
        return model

    def unload(self, name: str) -> bool:
        """Unload a model from memory."""
        if name in self._loaded_models:
            del self._loaded_models[name]
            return True
        return False

    def unload_all(self) -> None:
        """Unload all models."""
        self._loaded_models.clear()

    def is_loaded(self, name: str) -> bool:
        return name in self._loaded_models

    def get_loaded(self, name: str) -> Optional[Any]:
        return self._loaded_models.get(name)


# =============================================================================
# ModelCreator: Create new specialized models
# =============================================================================

class ModelCreator:
    """Creates new specialized models and registers them in catalog."""

    def __init__(self, catalog: ModelCatalog, models_dir: str = "models",
                 tokenizer_manager=None, default_config: Optional[Dict] = None):
        self.catalog = catalog
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.tokenizer_manager = tokenizer_manager
        self.default_config = default_config or {}

    def create_model(self, name: str, domain_keywords: List[str],
                     base_config: Optional[Dict] = None,
                     model_factory=None) -> ModelSpec:
        """Create a new specialized model.

        Args:
            name: Unique model name
            domain_keywords: List of domain keywords for routing
            base_config: Optional config overrides
            model_factory: Callable(config) -> model instance. If None, uses TinyLM.

        Returns:
            ModelSpec of the created model
        """
        if self.catalog.get(name) is not None:
            raise ValueError(f"Model '{name}' already exists")

        # Merge configs
        config = self.default_config.copy()
        if base_config:
            config.update(base_config)

        # Ensure vocab_size matches tokenizer if available
        if self.tokenizer_manager is not None:
            config.setdefault("vocab_size", self.tokenizer_manager.vocab_size)

        # Create model
        if model_factory is None:
            from ai import TinyLM
            # Filter config to TinyLM params
            model = TinyLM(
                vocab_size=config.get("vocab_size", 1000),
                ctx=config.get("ctx", 50),
                emb=config.get("emb", 50),
                hidden=config.get("hidden", 50),
            )
        else:
            model = model_factory(config)

        # Count params
        from ai import count_params
        param_count = count_params(model)

        # Save model
        model_path = self.models_dir / f"{name}.tlm"
        from ai import save_model
        save_model(model, str(model_path))

        # Create spec
        spec = ModelSpec(
            name=name,
            path=str(model_path),
            domain_keywords=domain_keywords,
            param_count=param_count,
            config=model.get_config(),
            created_at=time.time(),
            metrics={}
        )

        # Register in catalog
        self.catalog.add(spec)

        return spec


# =============================================================================
# ModelUpdater: Hot-swap model config, preserve weights where shapes match
# =============================================================================

class ModelUpdater:
    """Updates model configuration, preserving weights where shapes match."""

    def __init__(self, catalog: ModelCatalog, loader: ModelLoader):
        self.catalog = catalog
        self.loader = loader

    def update_model(self, name: str, new_config: Dict[str, Any]) -> ModelSpec:
        """Update model config, preserving weights where shapes match.

        Args:
            name: Model name in catalog
            new_config: New configuration dict

        Returns:
            Updated ModelSpec
        """
        spec = self.catalog.get(name)
        if spec is None:
            raise KeyError(f"Model '{name}' not found in catalog")

        # Load current model
        model = self.loader.load_model(name, force_reload=True)

        # Get current tensors
        old_tensors = model.get_tensors()

        # Create new model with new config
        new_config_full = {**spec.config, **new_config}
        new_config_full["arch"] = spec.config.get("arch", "TinyLM")

        if model_factory := self._get_factory(spec.config.get("arch")):
            new_model = model_factory(**{k: v for k, v in new_config_full.items() if k != "arch"})
        else:
            from ai import TinyLM
            new_model = TinyLM(**{k: v for k, v in new_config_full.items() if k != "arch"})

        # Copy weights where shapes match
        new_tensors = new_model.get_tensors()
        for tensor_name, new_rows in new_tensors.items():
            if tensor_name in old_tensors:
                old_rows = old_tensors[tensor_name]
                self._copy_matching_weights(old_rows, new_rows)

        # Save updated model
        from ai import save_model
        save_model(new_model, spec.path)

        # Update spec
        spec.config = new_model.get_config()
        spec.param_count = sum(len(row) for rows in new_tensors.values() for row in rows)
        self.catalog._save()

        # Reload in loader
        self.loader.unload(name)
        self.loader.load_model(name)

        return spec

    def _get_factory(self, arch: str):
        try:
            from registry import get_model_factory
            return get_model_factory(arch)
        except (ImportError, KeyError):
            return None

    def _copy_matching_weights(self, old_rows: List[List], new_rows: List[List]) -> None:
        """Copy weights from old to new where shapes match."""
        for i, (old_row, new_row) in enumerate(zip(old_rows, new_rows)):
            if len(old_row) == len(new_row):
                for j, (old_v, new_v) in enumerate(zip(old_row, new_row)):
                    new_v.data = old_v.data


# =============================================================================
# Catalog CLI (for command-line usage)
# =============================================================================

def catalog_cli():
    """Command-line interface for model catalog."""
    import sys

    catalog = ModelCatalog()

    if len(sys.argv) < 2:
        print("Usage: python -m model_mgmt <command> [args]")
        print("Commands: list, add, remove, info, find, discover")
        return

    cmd = sys.argv[1]

    if cmd == "list":
        for spec in catalog.list():
            print(f"{spec.name:20s} {spec.param_count:>10,} params  domains: {spec.domain_keywords}")

    elif cmd == "add":
        if len(sys.argv) < 5:
            print("Usage: add <name> <path> <param_count> <domain_keywords...>")
            return
        name = sys.argv[2]
        path = sys.argv[3]
        param_count = int(sys.argv[4])
        domain_keywords = sys.argv[5:] if len(sys.argv) > 5 else []

        # Load config from model file
        from ai import load_model
        model = load_model(path)
        config = model.get_config()

        spec = ModelSpec(
            name=name, path=path, domain_keywords=domain_keywords,
            param_count=param_count, config=config
        )
        catalog.add(spec)
        print(f"Added {name}")

    elif cmd == "remove":
        if len(sys.argv) < 3:
            print("Usage: remove <name>")
            return
        if catalog.remove(sys.argv[2]):
            print(f"Removed {sys.argv[2]}")
        else:
            print(f"Not found: {sys.argv[2]}")

    elif cmd == "info":
        if len(sys.argv) < 3:
            print("Usage: info <name>")
            return
        spec = catalog.get(sys.argv[2])
        if spec:
            print(json.dumps(spec.to_dict(), indent=2))
        else:
            print(f"Not found: {sys.argv[2]}")

    elif cmd == "find":
        if len(sys.argv) < 3:
            print("Usage: find <keyword>")
            return
        results = catalog.find_by_domain([sys.argv[2]])
        for spec in results:
            print(f"{spec.name:20s} {spec.param_count:>10,} params  domains: {spec.domain_keywords}")

    elif cmd == "discover":
        models_dir = sys.argv[2] if len(sys.argv) > 2 else "models"
        discovered = catalog.discover_models(models_dir)
        print(f"Discovered {len(discovered)} new model(s)")
        for spec in discovered:
            print(f"  {spec.name}: {spec.param_count:,} params, domains: {spec.domain_keywords}")


# Global catalog instance
MODEL_CATALOG = ModelCatalog()

if __name__ == "__main__":
    # Demo
    import tempfile
    from ai import TinyLM, save_model, count_params

    with tempfile.TemporaryDirectory() as tmpdir:
        catalog = ModelCatalog(f"{tmpdir}/catalog.json")
        creator = ModelCreator(catalog, f"{tmpdir}/models")

        # Create models
        spec1 = creator.create_model("tiny-code", ["code", "python"],
                                     base_config={"ctx": 100, "emb": 64, "hidden": 128})
        spec2 = creator.create_model("tiny-prose", ["text", "writing"],
                                     base_config={"ctx": 50, "emb": 50, "hidden": 50})

        print("Created:", spec1.name, spec1.param_count, "params")
        print("Created:", spec2.name, spec2.param_count, "params")

        # List
        print("\nCatalog:")
        for s in catalog.list():
            print(f"  {s.name}: {s.param_count:,} params, domains: {s.domain_keywords}")

        # Find by domain
        print("\nFind 'code':")
        for s in catalog.find_by_domain(["code"]):
            print(f"  {s.name}")

        # Loader
        loader = ModelLoader(catalog)
        model = loader.load_model("tiny-code")
        print(f"\nLoaded: {type(model).__name__}, params: {count_params(model)}")

        # Update
        updater = ModelUpdater(catalog, loader)
        updater.update_model("tiny-code", {"hidden": 256})
        print(f"\nUpdated tiny-code hidden to 256, new params: {catalog.get('tiny-code').param_count}")