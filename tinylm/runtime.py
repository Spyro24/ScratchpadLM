"""Multi-model runtime: lazy router + specialized models, routing, generation."""
import threading
import time
from typing import List, Optional

from tinylm.generation import fast_logits, _sample_next_token
from tinylm.serialization import load_model

# =============================================================================
# Multi-Model Runtime: Loads router + specialized models, routes prompts
# =============================================================================

class ModelRuntime:
    """Multi-model runtime that loads a router and specialized models,
    routes prompts to appropriate models, and supports collaborative generation."""

    def __init__(self, router_path: str = "router.tlm",
                 models_dir: str = "models",
                 tokenizer=None,
                 max_loaded_models: int = 5,
                 model_ttl: float = 300.0):
        """Initialize the multi-model runtime.

        Args:
            router_path: Path to router model file
            models_dir: Directory containing specialized model .tlm files
            tokenizer: BPETokenizer instance (uses global if not provided)
            max_loaded_models: Maximum number of specialized models to keep in memory
            model_ttl: Time-to-live for loaded models in seconds (auto-unload after idle)
        """
        self.router_path = router_path
        self.models_dir = models_dir
        self.tokenizer = tokenizer
        self.max_loaded_models = max_loaded_models
        self.model_ttl = model_ttl
        
        # Lazy-loaded models
        self._router = None
        self._models = {}  # model_name -> model instance
        self._model_configs = {}  # model_name -> config dict
        self._model_last_access = {}  # model_name -> last access timestamp
        self._catalog = None
        self._cleanup_thread = None
        self._running = False

    def start(self):
        """Start the runtime (background cleanup thread)."""
        if self._running:
            return
        self._running = True
        self._cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self._cleanup_thread.start()

    def stop(self):
        """Stop the runtime."""
        self._running = False
        if self._cleanup_thread:
            self._cleanup_thread.join(timeout=2)

    def _cleanup_loop(self):
        """Background loop to unload idle models."""
        while self._running:
            self._unload_idle_models()
            time.sleep(10)  # Check every 10 seconds

    def _unload_idle_models(self):
        """Unload models that have been idle longer than TTL."""
        now = time.time()
        to_unload = []
        for name, last_access in self._model_last_access.items():
            if now - last_access > self.model_ttl:
                to_unload.append(name)
        
        # Also enforce max loaded models limit
        if len(self._models) > self.max_loaded_models:
            # Sort by last access, unload oldest
            sorted_models = sorted(self._model_last_access.items(), key=lambda x: x[1])
            excess = len(self._models) - self.max_loaded_models
            to_unload.extend([name for name, _ in sorted_models[:excess]])
        
        for name in set(to_unload):
            self.unload_model(name)

    def _mark_accessed(self, name: str):
        """Mark a model as recently accessed."""
        self._model_last_access[name] = time.time()

    @property
    def router(self):
        """Lazy-load router model."""
        if self._router is None:
            self._router = load_model(self.router_path)
        return self._router

    @property
    def catalog(self):
        """Lazy-load model catalog."""
        if self._catalog is None:
            from model_mgmt import MODEL_CATALOG
            self._catalog = MODEL_CATALOG
        return self._catalog

    def _get_tokenizer(self):
        """Get tokenizer (use provided or global)."""
        if self.tokenizer is not None:
            return self.tokenizer
        from registry import TOKENIZER_MANAGER
        return TOKENIZER_MANAGER.tokenizer

    def load_model(self, name: str):
        """Load a specialized model by name."""
        if name in self._models:
            self._mark_accessed(name)
            return self._models[name]
        
        spec = self.catalog.get(name)
        if not spec:
            raise KeyError(f"Model '{name}' not found in catalog")
        
        model = load_model(spec.path)
        self._models[name] = model
        self._model_configs[name] = spec.config
        self._mark_accessed(name)
        return model

    def unload_model(self, name: str) -> bool:
        """Unload a model from memory."""
        if name in self._models:
            del self._models[name]
            self._model_last_access.pop(name, None)
            return True
        return False

    def unload_all(self):
        """Unload all specialized models (keeps router)."""
        self._models.clear()
        self._model_configs.clear()

    def list_loaded(self) -> List[str]:
        """List names of loaded specialized models."""
        return list(self._models.keys())

    def list_available(self) -> List[str]:
        """List all available models in catalog."""
        return [spec.name for spec in self.catalog.list() if spec.name != "router"]

    def route(self, prompt: str, max_subtasks: int = 4) -> List[dict]:
        """Route prompt to appropriate model(s) using router + catalog."""
        tokenizer = self._get_tokenizer()
        return self.router.route_multi(prompt, tokenizer, self.catalog, max_subtasks)

    def generate(self, prompt: str, max_tokens: int = 60, min_tokens: int = 0, temperature: float = 0.8,
                 top_k: int = 0, top_p: float = 0.0, repetition_penalty: float = 1.0,
                 stop_token_ids=None, stop_strings=None,
                 max_models: int = 1, stream_callback=None, collaborative: bool = False) -> str:
        """Generate text using routed model(s) with autostop support.

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate (default 60)
            min_tokens: Minimum tokens before autostop can trigger (default 0)
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Top-p (nucleus) sampling parameter
            repetition_penalty: Repetition penalty
            stop_token_ids: Optional set/list of token IDs that trigger stop
            stop_strings: Optional list of strings that trigger stop when generated
            max_models: Maximum number of models to use (1 = single, >1 = collaborative)
            stream_callback: Optional callback(token, model_name) for streaming
            collaborative: Force collaborative mode even with max_models=1

        Returns:
            Generated text
        """
        # Backward compatibility
        if 'n_tokens' in locals():
            pass  # max_tokens is the primary parameter now
        
        if max_models > 1 or collaborative:
            return self._generate_collaborative(prompt, max_tokens, temperature, top_k, top_p,
                                               repetition_penalty, max_models, stream_callback,
                                               min_tokens, stop_token_ids, stop_strings)
        else:
            routes = self.route(prompt, max_subtasks=1)
            if not routes:
                raise RuntimeError("No models available for generation")
            
            model_name = routes[0]["model"]
            model = self.load_model(model_name)
            tokenizer = self._get_tokenizer()
            
            # Generate with streaming support
            return self._generate_streaming(model, tokenizer, prompt, max_tokens, temperature,
                                           top_k, top_p, repetition_penalty, stream_callback, model_name,
                                           min_tokens, stop_token_ids, stop_strings)

    def _generate_streaming(self, model, tokenizer, prompt, max_tokens, temperature,
                           top_k, top_p, repetition_penalty, stream_callback, model_name,
                           min_tokens=0, stop_token_ids=None, stop_strings=None):
        """Generate with optional streaming callback."""
        ids = tokenizer.encode(prompt) or [0]
        model_fast_logits = getattr(model, 'fast_logits', None)
        
        # Convert stop_token_ids to set for O(1) lookup
        stop_tokens = set(stop_token_ids) if stop_token_ids else None
        
        # Pre-encode stop strings to token sequences
        stop_sequences = []
        if stop_strings:
            for s in stop_strings:
                seq = tokenizer.encode(s)
                if seq:
                    stop_sequences.append(seq)
        
        generated_text = ""
        start_time = time.time()
        
        for i in range(max_tokens):
            window = ids[-model.ctx:]
            window = [0] * (model.ctx - len(window)) + window
            if model_fast_logits is not None:
                logits = model_fast_logits(window)
            else:
                logits = fast_logits(model, window)
            next_id, should_stop = _sample_next_token(logits, temperature, top_k, top_p, 
                                                       repetition_penalty, ids, stop_tokens)
            ids.append(next_id)
            
            # Check stop strings
            if stop_sequences:
                for seq in stop_sequences:
                    if len(ids) >= len(seq) and ids[-len(seq):] == seq:
                        should_stop = True
                        break
            
            # Decode the new token
            new_token = tokenizer.decode([next_id])
            generated_text += new_token
            
            if stream_callback:
                stream_callback(new_token, model_name)
            
            # Check stop conditions
            if should_stop and i + 1 >= min_tokens:
                break
        
        return generated_text

    def _generate_collaborative(self, prompt, max_tokens, temperature, top_k, top_p,
                               repetition_penalty, max_models, stream_callback,
                               min_tokens=0, stop_token_ids=None, stop_strings=None):
        """Collaborative generation with multiple models, alternating between them."""
        routes = self.route(prompt, max_subtasks=max_models)
        if not routes:
            raise RuntimeError("No models available for generation")
        
        # Limit to available models
        available_routes = [r for r in routes if self.catalog.get(r["model"])]
        if not available_routes:
            raise RuntimeError("No routed models found in catalog")
        
        # Limit to max_models
        routes = available_routes[:max_models]
        
        tokenizer = self._get_tokenizer()
        model_names = [r["model"] for r in routes]
        
        # Pre-load all models
        models = {}
        for route in routes:
            models[route["model"]] = self.load_model(route["model"])
        
        # Determine tokens per model (round-robin distribution)
        tokens_per_model = max(1, max_tokens // len(routes))
        remaining = max_tokens
        
        generated_text = ""
        current_model_idx = 0
        
        if stream_callback:
            stream_callback(f"[Collaborative: {', '.join(model_names)}] ", "system")
        
        while remaining > 0:
            model_name = model_names[current_model_idx]
            model = models[model_name]
            
            # Generate tokens for this model
            chunk_size = min(tokens_per_model, remaining)
            
            if stream_callback:
                stream_callback(f"\n[{model_name}] ", "system")
            
            chunk_text = self._generate_streaming(
                model, tokenizer, prompt + generated_text, chunk_size, temperature,
                top_k, top_p, repetition_penalty, 
                lambda token: stream_callback(token, model_name) if stream_callback else None,
                model_name,
                min_tokens, stop_token_ids, stop_strings
            )
            
            generated_text += chunk_text
            remaining -= chunk_size
            current_model_idx = (current_model_idx + 1) % len(routes)
            
            # Update prompt context for next model
            # (In a full implementation, we'd pass the full context)
        
        return generated_text

    def generate_with_preview(self, prompt: str, max_tokens: int = 60, min_tokens: int = 0, 
                              temperature: float = 0.8, top_k: int = 0, top_p: float = 0.0,
                              repetition_penalty: float = 1.0, max_models: int = 1,
                              stop_token_ids=None, stop_strings=None,
                              preview_callback=None) -> dict:
        """Generate text with live token preview and timing stats.

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate (default 60)
            min_tokens: Minimum tokens before autostop can trigger (default 0)
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Top-p (nucleus) sampling parameter
            repetition_penalty: Repetition penalty
            max_models: Maximum number of models to use
            stop_token_ids: Optional set/list of token IDs that trigger stop
            stop_strings: Optional list of strings that trigger stop when generated
            preview_callback: Callback(token, model_name, token_idx, total_tokens, tokens_per_sec)

        Returns:
            Dict with 'text', 'tokens_generated', 'elapsed_sec', 'tokens_per_sec', 'model_used'
        """
        import time
        
        routes = self.route(prompt, max_subtasks=max_models)
        if not routes:
            raise RuntimeError("No models available for generation")
        
        model_name = routes[0]["model"]
        model = self.load_model(model_name)
        tokenizer = self._get_tokenizer()
        
        ids = tokenizer.encode(prompt) or [0]
        model_fast_logits = getattr(model, 'fast_logits', None)
        
        # Convert stop_token_ids to set for O(1) lookup
        stop_tokens = set(stop_token_ids) if stop_token_ids else None
        
        # Pre-encode stop strings to token sequences
        stop_sequences = []
        if stop_strings:
            for s in stop_strings:
                seq = tokenizer.encode(s)
                if seq:
                    stop_sequences.append(seq)
        
        generated_tokens = []
        generated_text = ""
        start_time = time.time()
        last_report_time = start_time
        
        for i in range(max_tokens):
            window = ids[-model.ctx:]
            window = [0] * (model.ctx - len(window)) + window
            if model_fast_logits is not None:
                logits = model_fast_logits(window)
            else:
                logits = fast_logits(model, window)
            next_id, should_stop = _sample_next_token(logits, temperature, top_k, top_p, 
                                                       repetition_penalty, ids, stop_tokens)
            ids.append(next_id)
            generated_tokens.append(next_id)
            
            # Check stop strings
            if stop_sequences:
                for seq in stop_sequences:
                    if len(ids) >= len(seq) and ids[-len(seq):] == seq:
                        should_stop = True
                        break
            
            # Decode the new token
            new_token = tokenizer.decode([next_id])
            generated_text += new_token
            
            # Call preview callback with timing info
            if preview_callback:
                elapsed = time.time() - start_time
                tps = (i + 1) / elapsed if elapsed > 0 else 0
                preview_callback(new_token, model_name, i + 1, max_tokens, tps)
            
            # Print progress every 10 tokens or at end
            if (i + 1) % 10 == 0 or (i + 1) == max_tokens:
                elapsed = time.time() - start_time
                tps = (i + 1) / elapsed if elapsed > 0 else 0
                print(f"  [{model_name}] Generated {i+1}/{max_tokens} tokens ({tps:.1f} tok/s)")
            
            # Check stop conditions
            if should_stop and i + 1 >= min_tokens:
                break
        
        total_elapsed = time.time() - start_time
        
        return {
            "text": generated_text,
            "tokens_generated": len(generated_tokens),
            "elapsed_sec": total_elapsed,
            "tokens_per_sec": len(generated_tokens) / total_elapsed if total_elapsed > 0 else 0,
            "model_used": model_name,
            "tokens": generated_tokens
        }

