"""ScratchpadLM entry point.

Usage:
    python3 main.py          # launch Tkinter GUI (default)
    python3 main.py --cli    # launch CLI (roadmap, not yet implemented)
    python3 main.py --mcp    # launch MCP system (roadmap, not yet implemented)
"""

import argparse
import sys


def run_gui():
    """Boot the Tkinter UI (must run on the main thread)."""
    import tkinter as tk

    from ui import App

    root = tk.Tk()
    App(root)
    root.mainloop()


def run_cli():
    """Launch the command-line interface (roadmap section 12)."""
    raise SystemExit("CLI is not implemented yet — see ROADMAP.md section 12 (CLI / API).")


def run_mcp():
    """Launch the MCP system (roadmap section 11, MCP support)."""
    raise SystemExit("MCP system is not implemented yet — see ROADMAP.md section 11 (Harness: MCP support).")


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="ScratchpadLM entry point (GUI by default).",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--cli", action="store_true", help="start the CLI instead of the GUI")
    group.add_argument("--mcp", action="store_true", help="start the MCP system instead of the GUI")
    args = parser.parse_args(argv)

    if args.cli:
        run_cli()
    elif args.mcp:
        run_mcp()
    else:
        run_gui()


if __name__ == "__main__":
    main()
