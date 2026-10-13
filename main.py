"""ScratchpadLM entry point.

Usage:
    python3 main.py          # launch Tkinter GUI (default)
    python3 main.py --cli    # launch CLI (roadmap, not yet implemented)
    python3 main.py --mcp    # launch MCP system (roadmap, not yet implemented)
"""

import argparse
import sys
import os


def run_gui():
    """Boot the Tkinter UI (must run on the main thread)."""
    import tkinter as tk

    from ui import App

    root = tk.Tk()
    App(root)
    root.mainloop()


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
    else:
        parser.print_help()
        sys.exit(1)


def cmd_router_train(args):
    """Train router model on corpus."""
    from ai import Router, RouterConfig, train_router, generate_router_training_data
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
    from ai import save_model
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


def cmd_generate(args):
    """Generate text from prompt."""
    from ai import generate, load_model
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
    from ai import train, load_model, save_model, sync_vocab
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

    ids = tokenizer.encode(data)
    if len(ids) <= model.ctx:
        raise ValueError(f"Need more than {model.ctx} tokens, got {len(ids)}")

    def progress_cb(done, total, loss, val_loss=None, lr=None):
        if done % 50 == 0 or done == total:
            vloss = f" val_loss={val_loss:.4f}" if val_loss else ""
            print(f"  Step {done}/{total} loss={loss:.4f}{vloss} lr={lr:.6f}")

    train(model, ids, steps=args.steps, lr=args.lr, progress_cb=progress_cb)
    save_model(model, "tinyLM" if not args.model else f"models/{args.model}.tlm")
    print("Training complete")


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
