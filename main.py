"""ScratchpadLM entry point.

Usage:
    python3 main.py          # launch Tkinter GUI (default)
    python3 main.py --cli    # launch CLI (roadmap, not yet implemented)
    python3 main.py --mcp    # launch MCP system (roadmap, not yet implemented)
"""

import argparse
import sys
import os
import time

# For taskpad CLI
try:
    from scratchpad import TaskType
except ImportError:
    TaskType = None


def run_gui():
    """Boot the GUI (must run on the main thread).

    PySide6 (Qt6) is the primary toolkit; Tkinter (stdlib) is the fallback
    when Qt is unavailable. Force one via SCRATCHPADLM_UI=pyside6|tkinter.
    """
    from ui_base import run_gui as _run_gui

    return _run_gui()


def run_cli(argv=None):
    """Launch the command-line interface."""
    if argv is None:
        argv = sys.argv[1:]
    parser = argparse.ArgumentParser(
        prog="tinylm",
        description="ScratchpadLM command-line interface",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Router training command
    router_parser = subparsers.add_parser("router", help="Router model commands")
    router_subparsers = router_parser.add_subparsers(dest="router_command", help="Router subcommands")

    train_parser = router_subparsers.add_parser("train", help="Pretrain router on corpus")
    train_parser.add_argument("--corpus", required=True, help="Path to training corpus file")
    train_parser.add_argument("--output", default="router.tlm", help="Output model path")
    train_parser.add_argument("--vocab-size", type=int, default=2000, help="Vocabulary size")
    train_parser.add_argument("--ctx", type=int, default=128, help="Context window size")
    train_parser.add_argument("--emb", type=int, default=64, help="Embedding dimension")
    train_parser.add_argument("--hidden", type=int, default=128, help="Hidden dimension")
    train_parser.add_argument("--steps", type=int, default=1000, help="Training steps")
    train_parser.add_argument("--lr", type=float, default=0.05, help="Learning rate")
    train_parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    train_parser.add_argument("--n-negatives", type=int, default=256, help="Negative samples")

    # Model management commands
    model_parser = subparsers.add_parser("model", help="Model management commands")
    model_subparsers = model_parser.add_subparsers(dest="model_command", help="Model subcommands")

    model_subparsers.add_parser("list", help="List all models in catalog")

    info_parser = model_subparsers.add_parser("info", help="Show model info")
    info_parser.add_argument("name", help="Model name")

    create_parser = model_subparsers.add_parser("create", help="Create new specialized model")
    create_parser.add_argument("name", help="Model name")
    create_parser.add_argument("--domains", required=True, help="Comma-separated domain keywords")
    create_parser.add_argument("--config", help="JSON config overrides")

    # Autotrain commands
    autotrain_parser = subparsers.add_parser("autotrain", help="Auto-training system commands")
    autotrain_subparsers = autotrain_parser.add_subparsers(dest="autotrain_command", help="Autotrain subcommands")

    autotrain_subparsers.add_parser("status", help="Show autotrain status")
    autotrain_subparsers.add_parser("enable", help="Enable auto-training")
    autotrain_subparsers.add_parser("disable", help="Disable auto-training")

    # Low-rating handler commands
    low_parser = autotrain_subparsers.add_parser("low-rated", help="List low-rated entries")
    low_parser.add_argument("--model", help="Filter by model name")
    low_parser.add_argument("--flag", action="store_true", help="Flag for review")

    review_parser = autotrain_subparsers.add_parser("review", help="Show flagged entries for review")
    review_parser.add_argument("--model", help="Filter by model name")

    avoid_parser = autotrain_subparsers.add_parser("avoidance", help="Run avoidance training on low-rated entries")
    avoid_parser.add_argument("--model", required=True, help="Model name to train")
    avoid_parser.add_argument("--threshold", type=int, default=2, help="Low threshold (default: 2)")

    config_parser = autotrain_subparsers.add_parser("config", help="Configure auto-training")
    config_parser.add_argument("--threshold", type=int, help="High rating threshold (1-5)")
    config_parser.add_argument("--batch", type=int, help="Min batch size")

    # Generate command
    gen_parser = subparsers.add_parser("generate", help="Generate text from prompt")
    gen_parser.add_argument("prompt", help="Input prompt")
    gen_parser.add_argument("--model", help="Model name (uses current model if not specified)")
    gen_parser.add_argument("--tokens", type=int, default=60, help="Number of tokens to generate")
    gen_parser.add_argument("--temperature", type=float, default=0.8, help="Temperature")
    gen_parser.add_argument("--top-k", type=int, default=0, help="Top-k sampling")
    gen_parser.add_argument("--top-p", type=float, default=0.0, help="Top-p sampling")

    # Train command (general)
    train_parser = subparsers.add_parser("train", help="Train a model on data")
    train_parser.add_argument("data", help="Training text or file path")
    train_parser.add_argument("--model", help="Model name (uses current model if not specified)")
    train_parser.add_argument("--steps", type=int, default=1000, help="Training steps")
    train_parser.add_argument("--lr", type=float, default=0.05, help="Learning rate")
    train_parser.add_argument("--no-markers", dest="markers", action="store_false",
                              help="Skip [EOL]/[EOS]/[EOT] structure markers")

    # Eval command
    eval_parser = subparsers.add_parser("eval", help="Evaluate a model checkpoint")
    eval_parser.add_argument("--checkpoint", required=True, help="Path to checkpoint file")
    eval_parser.add_argument("--data", required=True, help="Validation data file")
    eval_parser.add_argument("--model", help="Model name (loads from checkpoint if not specified)")
    eval_parser.add_argument("--batch-size", type=int, default=16, help="Batch size for evaluation")
    eval_parser.add_argument("--n-negatives", type=int, default=256, help="Negative samples for sampled softmax")

    # Serve command
    serve_parser = subparsers.add_parser("serve", help="Start HTTP server for generation/embedding/routing")
    serve_parser.add_argument("--port", type=int, default=8080, help="Server port")
    serve_parser.add_argument("--host", default="127.0.0.1", help="Server host")
    serve_parser.add_argument("--model", help="Default model to serve")

    # Interactive prompt command (alias for generate without subcommand)
    # This allows: tinylm "your prompt here"

    # Scratchpad command
    scratchpad_parser = subparsers.add_parser("scratchpad", help="Scratchpad commands")
    scratchpad_subparsers = scratchpad_parser.add_subparsers(dest="scratchpad_command", help="Scratchpad subcommands")
    scratchpad_subparsers.add_parser("show", help="Show scratchpad contents")
    scratchpad_subparsers.add_parser("clear", help="Clear scratchpad")
    add_parser = scratchpad_subparsers.add_parser("add", help="Add entry to scratchpad")
    add_parser.add_argument("--thought", required=True, help="Thought text")
    add_parser.add_argument("--result", required=True, help="Result text")
    add_parser.add_argument("--model", required=True, help="Model name")

    # Taskpad command
    taskpad_parser = subparsers.add_parser("taskpad", help="Taskpad commands")
    taskpad_subparsers = taskpad_parser.add_subparsers(dest="taskpad_command", help="Taskpad subcommands")
    taskpad_subparsers.add_parser("list", help="List all tasks")
    
    task_create = taskpad_subparsers.add_parser("create", help="Create new task")
    task_create.add_argument("--type", required=True, choices=[t.value for t in TaskType], help="Task type")
    task_create.add_argument("--prompt", required=True, help="Task prompt")
    task_create.add_argument("--model", help="Model name")
    task_create.add_argument("--deps", nargs="*", default=[], help="Dependency task IDs")
    
    task_execute = taskpad_subparsers.add_parser("execute", help="Execute a task")
    task_execute.add_argument("task_id", help="Task ID to execute")
    
    taskpad_subparsers.add_parser("run-all", help="Execute all pending tasks")

    if len(argv) == 0:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args(argv)

    if args.command == "router" and args.router_command == "train":
        cmd_router_train(args)
    elif args.command == "model":
        cmd_model(args)
    elif args.command == "autotrain":
        cmd_autotrain(args)
    elif args.command == "generate":
        cmd_generate(args)
    elif args.command == "train":
        cmd_train(args)
    elif args.command == "eval":
        cmd_eval(args)
    elif args.command == "serve":
        cmd_serve(args)
    elif args.command == "scratchpad":
        cmd_scratchpad(args)
    elif args.command == "taskpad":
        cmd_taskpad(args)
    else:
        parser.print_help()
        sys.exit(1)


def cmd_router_train(args):
    """Train router model on corpus."""
    from tinylm import Router, RouterConfig, train_router, generate_router_training_data
    from tokenizer import BPETokenizer
    import time

    print(f"Loading corpus from {args.corpus}...")
    with open(args.corpus, "r", encoding="utf-8") as f:
        corpus = f.read()
    print(f"Corpus size: {len(corpus):,} characters")

    # Create/load tokenizer
    tokenizer = BPETokenizer()
    tokenizer_path = "bpe_merges.txt"
    if os.path.exists(tokenizer_path):
        tokenizer.load(tokenizer_path)
        print(f"Loaded tokenizer with {len(tokenizer.vocab)} tokens")
    else:
        print("Training new tokenizer...")
        tokenizer.train(corpus, vocab_size=args.vocab_size)
        tokenizer.save(tokenizer_path)
        print(f"Tokenizer trained: {len(tokenizer.vocab)} tokens")

    # Generate router training data
    print("Generating router training data...")
    examples = generate_router_training_data(corpus, tokenizer, args.vocab_size, args.ctx)
    print(f"Generated {len(examples)} training examples")

    # Create router model
    config = RouterConfig(
        vocab_size=len(tokenizer.vocab),
        ctx=args.ctx,
        emb=args.emb,
        hidden=args.hidden,
        n_layers=2,
        d_ff=512
    )
    router = Router(config, seed=42)
    print(f"Router model: {router.count_params():,} params" if hasattr(router, 'count_params') else "Router model created")

    # Train
    print(f"Training router for {args.steps} steps...")
    start = time.time()

    def progress_cb(done, total, loss, val_loss=None, lr=None):
        if done % 50 == 0 or done == total:
            vloss = f" val_loss={val_loss:.4f}" if val_loss else ""
            print(f"  Step {done}/{total} loss={loss:.4f}{vloss} lr={lr:.6f}")

    train_router(router, tokenizer, examples,
                 steps=args.steps, lr=args.lr, batch_size=args.batch_size,
                 n_negatives=args.n_negatives, progress_cb=progress_cb)

    elapsed = time.time() - start
    print(f"Training completed in {elapsed:.1f}s")

    # Save
    from tinylm import save_model
    save_model(router, args.output)
    print(f"Router saved to {args.output}")


def cmd_model(args):
    """Model management commands."""
    from model_mgmt import ModelCatalog, ModelCreator, MODEL_CATALOG
    import json

    catalog = MODEL_CATALOG

    if args.model_command == "list":
        specs = catalog.list()
        if not specs:
            print("No models in catalog")
            return
        print(f"{'Name':<20} {'Params':>10}  Domains")
        print("-" * 60)
        for spec in specs:
            domains = ", ".join(spec.domain_keywords)
            print(f"{spec.name:<20} {spec.param_count:>10,}  {domains}")

    elif args.model_command == "info":
        spec = catalog.get(args.name)
        if not spec:
            print(f"Model '{args.name}' not found")
            return
        print(f"Name: {spec.name}")
        print(f"Path: {spec.path}")
        print(f"Params: {spec.param_count:,}")
        print(f"Domains: {spec.domain_keywords}")
        print(f"Config: {json.dumps(spec.config, indent=2)}")
        print(f"Metrics: {json.dumps(spec.metrics, indent=2)}")

    elif args.model_command == "create":
        domains = [d.strip() for d in args.domains.split(",") if d.strip()]
        config = {}
        if args.config:
            config = json.loads(args.config)

        creator = ModelCreator(catalog, "models")
        spec = creator.create_model(args.name, domains, base_config=config)
        print(f"Created model '{spec.name}' ({spec.param_count:,} params)")


def cmd_autotrain(args):
    """Auto-training commands."""
    from autotrain import create_autotrain_system, RatingCollector

    def mock_loader():
        class MockLoader:
            def load_model(self, name):
                return None
        return MockLoader()

    collector, auto_trainer = create_autotrain_system(mock_loader)

    if args.autotrain_command == "status":
        status = auto_trainer.get_system_status()
        print(f"Auto-trainer: {'Running' if status['autotrainer_running'] else 'Stopped'}")
        print(f"Global avg rating: {status['global_avg_rating']:.2f}")
        print(f"Total ratings: {status['total_ratings']}")
        print(f"Total train sessions: {status['total_train_sessions']}")
        print(f"Total pending feedback: {status['total_pending_feedback']}")
        print(f"Models needing attention: {status['models_needing_attention']}")
        print(f"Models count: {status['models_count']}")

    elif args.autotrain_command == "enable":
        auto_trainer.start()
        print("Auto-trainer enabled")

    elif args.autotrain_command == "disable":
        auto_trainer.stop()
        print("Auto-trainer disabled")

    elif args.autotrain_command == "config":
        if args.threshold is not None:
            collector.high_threshold = args.threshold
        if args.batch is not None:
            auto_trainer.min_batch = args.batch
        print(f"Updated: high_threshold={collector.high_threshold}, min_batch={auto_trainer.min_batch}")

    elif args.autotrain_command == "low-rated":
        entries = auto_trainer.handle_low_rated(args.model, auto_flag=args.flag)
        if not entries:
            print("No low-rated entries found")
            return
        print(f"Found {len(entries)} low-rated entries (rating ≤ {collector.low_threshold}):")
        for e in entries:
            flag = " [FLAGGED]" if e.metadata.get('flagged_for_review') else ""
            print(f"  [{e.model_name}] {e.rating}/5: {e.prompt[:60]}... -> {e.response[:60]}...{flag}")

    elif args.autotrain_command == "review":
        flagged = auto_trainer.flag_for_review(args.model)
        if not flagged:
            print("No entries flagged for review")
            return
        print(f"Flagged entries for review ({len(flagged)}):")
        for i, e in enumerate(flagged, 1):
            print(f"\n{i}. [{e['model_name']}] Rating: {e['rating']}/5")
            print(f"   Prompt: {e['prompt'][:80]}...")
            print(f"   Response: {e['response'][:80]}...")
            print(f"   Time: {time.ctime(e['timestamp'])}")

    elif args.autotrain_command == "avoidance":
        if not args.model:
            print("Error: --model is required for avoidance training")
            return
        entries = auto_trainer.handle_low_rated(args.model)
        if not entries:
            print("No low-rated entries found for this model")
            return
        print(f"Running avoidance training on {len(entries)} low-rated entries...")
        result = auto_trainer.train_avoidance(args.model, entries)
        if result.get("success"):
            print(f"Avoidance training complete: {result}")
        else:
            print(f"Avoidance training failed: {result.get('error')}")


def cmd_generate(args):
    """Generate text from prompt."""
    from tinylm import generate, load_model
    from registry import TOKENIZER_MANAGER

    tokenizer = TOKENIZER_MANAGER.tokenizer
    if os.path.exists("bpe_merges.txt"):
        tokenizer.load("bpe_merges.txt")

    if args.model:
        model = load_model(f"models/{args.model}.tlm")
    else:
        model = load_model("tinyLM")

    text = generate(model, tokenizer, args.prompt,
                    n_tokens=args.tokens, temperature=args.temperature,
                    top_k=args.top_k, top_p=args.top_p)
    print(text)


def cmd_train(args):
    """Train a model on data."""
    from tinylm import (load_model, save_model, sync_vocab, insert_markers,
                        encode_marked, render_marked, train_with_stats)
    from registry import TOKENIZER_MANAGER

    tokenizer = TOKENIZER_MANAGER.tokenizer
    if os.path.exists("bpe_merges.txt"):
        tokenizer.load("bpe_merges.txt")

    # Load training data
    if os.path.isfile(args.data):
        with open(args.data, "r", encoding="utf-8") as f:
            data = f.read()
    else:
        data = args.data

    if args.model:
        model = load_model(f"models/{args.model}.tlm")
    else:
        model = load_model("tinyLM")

    tokenizer.extend(data, vocab_size=2000)
    tokenizer.save("bpe_merges.txt")
    sync_vocab(model, len(tokenizer.vocab))

    if getattr(args, "markers", True):
        marked = insert_markers(data)
        sample = marked[:600]
        print("Training text with markers:")
        print(render_marked(sample + ("..." if len(marked) > 600 else "")))
        ids = encode_marked(marked, tokenizer, len(tokenizer.vocab))
    else:
        ids = tokenizer.encode(data)
    if len(ids) <= model.ctx:
        raise ValueError(f"Need more than {model.ctx} tokens, got {len(ids)}")

    def progress_cb(done, total, loss, val_loss=None, lr=None):
        if done % 50 == 0 or done == total:
            vloss = f" val_loss={val_loss:.4f}" if val_loss else ""
            print(f"  Step {done}/{total} loss={loss:.4f}{vloss} lr={lr:.6f}")

    out_path = f"models/{args.model}.tlm" if args.model else "tinyLM"
    name = args.model or os.path.splitext(os.path.basename(out_path))[0]
    info = train_with_stats(name, model, ids, steps=args.steps, lr=args.lr,
                            progress_cb=progress_cb)
    if info["improved"]:
        save_model(model, out_path)
        print(f"Training complete: best loss {info['best_loss']:.4f}, model saved")
    else:
        print(f"Training complete: no improvement (best stays {info['prev_best']}) - "
              f"weights reverted, {out_path} kept")


def cmd_eval(args):
    """Evaluate a model checkpoint."""
    from tinylm import evaluate, load_checkpoint, load_model, resume_from_checkpoint
    from registry import TOKENIZER_MANAGER

    tokenizer = TOKENIZER_MANAGER.tokenizer
    if os.path.exists("bpe_merges.txt"):
        tokenizer.load("bpe_merges.txt")

    # Load validation data
    with open(args.data, "r", encoding="utf-8") as f:
        val_text = f.read()

    if args.model:
        model = load_model(f"models/{args.model}.tlm")
    else:
        # Load from checkpoint
        ckpt = load_checkpoint(args.checkpoint)
        model = load_model("tinyLM")  # Will be replaced by resume
        resume_from_checkpoint(model, ckpt)

    ids = tokenizer.encode(val_text)
    if len(ids) <= model.ctx:
        raise ValueError(f"Need more than {model.ctx} tokens, got {len(ids)}")

    print(f"Evaluating on {len(ids)} tokens...")
    perplexity = evaluate(model, ids, batch_size=args.batch_size, n_negatives=args.n_negatives)
    print(f"Perplexity: {perplexity:.4f}")


def cmd_serve(args):
    """Start HTTP server for generation/embedding/routing."""
    print(f"Starting server on {args.host}:{args.port}...")
    print("Note: HTTP server not yet implemented. See ROADMAP.md section 12 (CLI/API).")
    # TODO: Implement HTTP server with endpoints:
    # POST /generate - text generation
    # POST /embed - embeddings
    # POST /route - model routing
    # GET /health - health check


def cmd_prompt(args):
    """Interactive prompt - generate text from prompt (allows: tinylm "prompt")."""
    from tinylm import generate, load_model
    from registry import TOKENIZER_MANAGER

    tokenizer = TOKENIZER_MANAGER.tokenizer
    if os.path.exists("bpe_merges.txt"):
        tokenizer.load("bpe_merges.txt")

    model = load_model("tinyLM")
    text = generate(model, tokenizer, args.prompt, n_tokens=60, temperature=0.8)
    print(text)


def cmd_scratchpad(args):
    """Scratchpad commands."""
    from scratchpad import Scratchpad, TaskType
    
    scratchpad = Scratchpad()
    
    if args.scratchpad_command == "show":
        context = scratchpad.get_context(max_entries=50)
        if context:
            print(context)
        else:
            print("Scratchpad is empty")
    
    elif args.scratchpad_command == "clear":
        scratchpad.clear()
        print("Scratchpad cleared")
    
    elif args.scratchpad_command == "add":
        scratchpad.add(args.thought, args.result, args.model)
        print(f"Added entry for model '{args.model}'")


def _default_task_executors(catalog=None):
    """Generate-backed executors so `taskpad execute`/`run-all` work from the CLI.

    Models resolve ONLY through the catalog (no hardwired names or fallback
    files — only the `router` identity is fixed elsewhere): a task whose
    `model_name` is unlisted fails with a clear error. `max_tokens` may be
    overridden per task via `task.parameters`.
    """
    from scratchpad import TaskType
    from tinylm import generate, load_model
    from registry import TOKENIZER_MANAGER

    if catalog is None:
        from model_mgmt import MODEL_CATALOG as catalog
    tokenizer = TOKENIZER_MANAGER.tokenizer
    if os.path.exists("bpe_merges.txt"):
        tokenizer.load("bpe_merges.txt")

    def _run(task):
        spec = catalog.get(task.model_name) if task.model_name else None
        if spec is None:
            raise RuntimeError(
                f"Model '{task.model_name}' is not in the catalog")
        max_tokens = (task.parameters or {}).get("max_tokens", 60)
        return generate(load_model(spec.path), tokenizer, task.prompt,
                        max_tokens=max_tokens)

    return {t: _run for t in (TaskType.GENERATE, TaskType.CODE, TaskType.ANALYZE,
                              TaskType.REASON, TaskType.SEARCH, TaskType.CUSTOM)}


def cmd_taskpad(args):
    """Taskpad commands."""
    from scratchpad import TaskPad, TaskType

    taskpad = TaskPad()
    for task_type, executor in _default_task_executors().items():
        taskpad.register_executor(task_type, executor)
    
    if args.taskpad_command == "list":
        tasks = taskpad.list_tasks()
        if not tasks:
            print("No tasks")
            return
        print(f"{'ID':<30} {'Type':<12} {'Status':<12} {'Model':<15} {'Prompt'}")
        print("-" * 100)
        for t in tasks:
            prompt_preview = t.prompt[:50] + "..." if len(t.prompt) > 50 else t.prompt
            model = t.model_name or "-"
            print(f"{t.id:<30} {t.type.value:<12} {t.status.value:<12} {model:<15} {prompt_preview}")
    
    elif args.taskpad_command == "create":
        task = taskpad.create_task(
            TaskType(args.type),
            args.prompt,
            args.model,
            dependencies=args.deps
        )
        print(f"Created task: {task.id}")
    
    elif args.taskpad_command == "execute":
        if not args.task_id:
            print("Error: task_id required")
            return
        print(f"Executing task {args.task_id}...")
        try:
            task = taskpad.execute_task(args.task_id)
        except (KeyError, ValueError) as e:
            print(f"Error: {e}")
            return
        print(f"Status: {task.status.value}")
        if task.status.value == "completed":
            print(task.result)
        else:
            print(f"Error: {task.error}")

    elif args.taskpad_command == "run-all":
        print("Running all pending tasks...")
        executed = taskpad.execute_all_pending()
        print(f"Executed {len(executed)} tasks")
        for task in executed:
            state = task.status.value
            detail = task.result if state == "completed" else task.error
            preview = (detail[:80] + "...") if detail and len(detail) > 80 else (detail or "")
            print(f"  {task.id} [{state}] {preview}")


def run_mcp():
    """Launch the MCP system (roadmap section 11, MCP support)."""
    raise SystemExit("MCP system is not implemented yet — see ROADMAP.md section 11 (Harness: MCP support).")


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    
    # Check for --cli or --mcp flags first
    if "--cli" in argv:
        # Remove --cli and pass remaining args to CLI
        cli_argv = [a for a in argv if a != "--cli"]
        run_cli(cli_argv)
        return
    elif "--mcp" in argv:
        run_mcp()
        return
    elif len(argv) > 0 and not argv[0].startswith("-"):
        # Interactive prompt: tinylm "your prompt here"
        # Wrap as CLI generate command
        run_cli(["generate", argv[0]] + argv[1:])
        return
    
    # Default: run GUI
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="ScratchpadLM entry point (GUI by default).",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--cli", action="store_true", help="start the CLI instead of the GUI")
    group.add_argument("--mcp", action="store_true", help="start the MCP system instead of the GUI")
    parser.parse_args(argv)
    run_gui()


if __name__ == "__main__":
    main()
