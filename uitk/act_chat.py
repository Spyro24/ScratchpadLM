"""Tkinter UI mixin (stdlib fallback). Part of the `uitk/` package split from `ui.py`."""
import json
import os
import queue
import threading
import time
import tkinter as tk
from tkinter import scrolledtext, messagebox, ttk, filedialog
from pathlib import Path

from tokenizer import BPETokenizer
from ai import TinyLM, train, generate, load_model, save_model, sync_vocab, train_parallel, count_params, Router, RouterTrainingExample
from registry import MODEL_REGISTRY, TASK_REGISTRY, TOKENIZER_MANAGER, get_tokenizer
from model_mgmt import ModelCatalog, ModelCreator, ModelLoader, MODEL_CATALOG
from config import Config
from autotrain import RatingCollector, FeedbackEntry, create_autotrain_system
from ui_base import (
    CONFIG_PATH,
    MODEL_PATH,
    PALETTE,
    PALETTE_LIGHT,
    PALETTE_DARK,
    TOKENIZER_PATH,
    get_available_models,
    load_config,
    apply_tk_theme,
    get_palette,
)
from ui_base import parse_bool as _parse_bool
from ui_base import write_config_yaml as _write_config_yaml
from uitk.const import cfg, VOCAB_SIZE, CTX, EMB, HIDDEN


class ChatActionsMixin:
    def on_send_prompt(self):
        if self.busy:
            return
        prompt = self.prompt_var.get().strip()
        if not prompt:
            return
        # Clear prompt entry
        self.prompt_var.delete(0, "end")
        # Show user message immediately
        self._append_history("user", prompt)
        # Get generation settings
        try:
            max_tokens = int(self.max_tokens_var.get())
            min_tokens = int(self.min_tokens_var.get())
            temperature = float(self.temp_var.get())
            top_k = int(self.top_k_var.get())
            top_p = float(self.top_p_var.get())
            repetition_penalty = float(self.rep_penalty_var.get())
            stop_token_ids = [int(x.strip()) for x in self.stop_tokens_var.get().split(",") if x.strip()]
            stop_strings = [x.strip() for x in self.stop_strings_var.get().split(",") if x.strip()]
        except ValueError:
            max_tokens, min_tokens, temperature, top_k, top_p, repetition_penalty = 60, 0, 0.8, 0, 0.0, 1.0
            stop_token_ids = []
            stop_strings = []
        self._stream_active = False  # streamed tokens open a fresh "AI:" block
        # Snapshot model-selection state on the main thread (Tk vars must not
        # be read from background threads).
        use_selected = bool(self.use_selected_model.get())
        selected_name = self._model_name
        self._run_in_background("Generating", self._generate_job, prompt, max_tokens, min_tokens, temperature, top_k, top_p, repetition_penalty, stop_token_ids, stop_strings, use_selected, selected_name)

    def _generate_job(self, prompt, max_tokens, min_tokens, temperature, top_k, top_p, repetition_penalty, stop_token_ids, stop_strings, use_selected=False, selected_name=None):
        # Route the prompt using Router (or use the manually selected model)
        model, model_name = self._route_prompt(prompt, use_selected=use_selected, selected_name=selected_name)
        
        # Use streaming generation with callback for live updates
        def stream_callback(token_str, token_idx, total_tokens, tps):
            self.queue.put(("stream_token", token_str, (token_idx, total_tokens, tps)))
        
        # Collect routing data for Router training
        example = self._collect_routing_data(prompt, model_name, "")
        
        text = generate(model, self.tok, prompt, max_tokens=max_tokens, min_tokens=min_tokens, temperature=temperature, 
                        top_k=top_k, top_p=top_p, repetition_penalty=repetition_penalty,
                        stop_token_ids=stop_token_ids, stop_strings=stop_strings,
                        stream_callback=stream_callback)
        
        # Collect response for routing data
        if example:
            example.subtasks[0]["response"] = text
            # Store for Router training
            self._store_routing_example(example)
        
        # Log to unified log
        self.queue.put(("log", "generation", f"Prompt: {prompt[:100]}{'...' if len(prompt) > 100 else ''}"))
        self.queue.put(("log", "generation", f"Routed to: {model_name}"))
        self.queue.put(("log", "generation", f"Response: {text[:200]}{'...' if len(text) > 200 else ''}"))
        return prompt, text, model_name

    def _store_routing_example(self, example):
        """Store routing example for Router training."""
        try:
            # Save to a file for later Router training
            import json
            with open("router_training_data.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "prompt": example.prompt,
                    "subtasks": example.subtasks
                }) + "\n")
        except Exception:
            pass

    def _route_prompt(self, prompt, use_selected=False, selected_name=None):
        """Route prompt using the Router model (runs on a worker thread).

        When `use_selected` is True the manually selected model is used
        directly and the router is bypassed. `selected_name` must be
        snapshotted on the main thread by the caller.
        """
        current_name = selected_name or getattr(self, "_model_name", "TinyLM")

        # If user wants to use selected model, bypass router
        if use_selected:
            return self.model, current_name
        
        try:
            routes = self.router.route_multi(prompt, self.tok, MODEL_CATALOG, max_subtasks=3)
            if not routes:
                # Fallback to default model
                return self.model, current_name
            
            # Use the top route
            best_route = routes[0]
            model_name = best_route["model"]
            
            # Load the model if different from current
            if model_name != current_name:
                loader = ModelLoader(MODEL_CATALOG, tokenizer_manager=TOKENIZER_MANAGER)
                model = loader.load_model(model_name)
                return model, model_name
            return self.model, current_name
        except Exception as e:
            # Log via the queue (thread-safe) instead of touching widgets
            self.queue.put(("log", "error", f"Routing failed: {e}, falling back to current model"))
            return self.model, current_name

    def _collect_routing_data(self, prompt, model_name, response):
        """Collect routing training data from user interaction."""
        # This creates training examples for the Router
        # The Router learns to predict which model should handle which prompt
        try:
            example = RouterTrainingExample(
                prompt=prompt,
                subtasks=[{"model": model_name, "prompt": prompt}]
            )
            return example
        except Exception:
            return None

    def on_clear_history(self):
        self.history.configure(state="normal")
        self.history.delete("1.0", "end")
        self.history.configure(state="disabled")

    def on_generate_preview(self):
        """Generate with detailed preview showing tokens/sec."""
        if self.busy:
            return
        prompt = self.prompt_var.get().strip()
        if not prompt:
            return
        
        # Get generation settings
        try:
            max_tokens = int(self.max_tokens_var.get())
            min_tokens = int(self.min_tokens_var.get())
            temperature = float(self.temp_var.get())
            top_k = int(self.top_k_var.get())
            top_p = float(self.top_p_var.get())
            repetition_penalty = float(self.rep_penalty_var.get())
            stop_token_ids = [int(x.strip()) for x in self.stop_tokens_var.get().split(",") if x.strip()]
            stop_strings = [x.strip() for x in self.stop_strings_var.get().split(",") if x.strip()]
        except ValueError:
            max_tokens, min_tokens, temperature, top_k, top_p, repetition_penalty = 60, 0, 0.8, 0, 0.0, 1.0
            stop_token_ids = []
            stop_strings = []
        
        # Clear prompt entry
        self.prompt_var.delete(0, "end")
        
        # Show user message
        self._append_history("user", prompt)
        self._stream_active = False

        # Snapshot model-selection state on the main thread.
        use_selected = bool(self.use_selected_model.get())
        selected_name = self._model_name
        # Run in background with preview
        self._run_in_background("Generating (Preview)", self._generate_preview_job,
                               prompt, max_tokens, min_tokens, temperature, top_k, top_p, repetition_penalty, stop_token_ids, stop_strings, use_selected, selected_name)

    def _generate_preview_job(self, prompt, max_tokens, min_tokens, temperature, top_k, top_p, repetition_penalty, stop_token_ids, stop_strings, use_selected=False, selected_name=None):
        """Background job for generate_with_preview (always uses the selected model)."""
        from ai import generate_with_preview
        result = generate_with_preview(self.model, self.tok, prompt,
                                      max_tokens=max_tokens, min_tokens=min_tokens, temperature=temperature,
                                      top_k=top_k, top_p=top_p,
                                      repetition_penalty=repetition_penalty,
                                      stop_token_ids=stop_token_ids, stop_strings=stop_strings)
        return prompt, result

    # ---- admin actions -----------------------------------------------------
