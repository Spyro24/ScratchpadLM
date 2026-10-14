#!/usr/bin/env python3
"""Remove generated artifacts (stdlib only).

Default: build outputs (``*.so``) + Python bytecode caches. Never touches
weights, tokenizers, configs, training data, or chat/scratchpad history:

  protected: tinyLM*, bpe_merges.txt, models/, trainingsData/, *.tlm,
             config.yaml, tokenizer files, *.jsonl history/logs

Usage:
    python3 clean.py [--dry-run] [--checkpoints] [--stats] [--all]
"""
import argparse
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Never delete these, even with --all.
PROTECTED_NAMES = {}
PROTECTED_DIRS = {"models", "trainingsData"}
PROTECTED_SUFFIXES = {".tlm"}


def _is_protected(path: Path) -> bool:
    if path.name in PROTECTED_NAMES:
        return True
    if path.suffix in PROTECTED_SUFFIXES:
        return True
    return any(part in PROTECTED_DIRS for part in path.parts)


def collect(checkpoints: bool = False, stats: bool = False):
    """Return sorted list of artifact paths to remove."""
    found = []

    # Build outputs: shared libraries at root and in csrc/.
    for so in list(ROOT.glob("*.so")) + list((ROOT / "csrc").glob("*.so")):
        if not _is_protected(so):
            found.append(so)

    # Python bytecode caches (repo-wide, including skill helpers).
    for cache in ROOT.rglob("__pycache__"):
        found.append(cache)
    for pyc in ROOT.rglob("*.pyc"):
        if pyc.parent.name != "__pycache__":  # already covered via parent dir
            found.append(pyc)

    if checkpoints:
        ckpt = ROOT / "checkpoints"
        if ckpt.is_dir():
            found.extend(p for p in ckpt.iterdir())

    if stats:
        sp = ROOT / "training_stats.json"
        if sp.is_file():
            found.append(sp)
        best = ROOT / "best_models"
        if best.is_dir():
            found.extend(p for p in best.iterdir())

    # De-duplicate (a .pyc inside a listed __pycache__ is removed with it).
    uniq = []
    seen = set()
    for p in found:
        rp = str(p.resolve())
        if rp not in seen:
            seen.add(rp)
            uniq.append(p)
    return sorted(uniq)


def clean(paths, dry_run: bool = False) -> int:
    """Remove paths (files or dirs). Returns count removed."""
    removed = 0
    for p in paths:
        if _is_protected(p):
            print(f"  SKIP (protected): {p}")
            continue
        if dry_run:
            print(f"  would remove: {p.relative_to(ROOT) if p.is_relative_to(ROOT) else p}")
            continue
        try:
            if p.is_dir() and not p.is_symlink():
                shutil.rmtree(p)
            else:
                p.unlink(missing_ok=True)
            print(f"  removed: {p.relative_to(ROOT) if p.is_relative_to(ROOT) else p}")
            removed += 1
        except FileNotFoundError:
            pass
        except OSError as exc:
            print(f"  FAILED: {p}: {exc}", file=sys.stderr)
    return removed


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Remove generated artifacts.")
    ap.add_argument("--dry-run", action="store_true",
                    help="print what would be removed, remove nothing")
    ap.add_argument("--checkpoints", action="store_true",
                    help="also empty checkpoints/")
    ap.add_argument("--stats", action="store_true",
                    help="also remove training_stats.json and best_models/ contents")
    ap.add_argument("--all", action="store_true",
                    help="shorthand for --checkpoints --stats")
    args = ap.parse_args(argv)

    do_ckpt = args.checkpoints or args.all
    do_stats = args.stats or args.all
    paths = collect(checkpoints=do_ckpt, stats=do_stats)
    if not paths:
        print("Nothing to clean.")
        return 0
    print(f"{'Would remove' if args.dry_run else 'Removing'} {len(paths)} path(s):")
    n = clean(paths, dry_run=args.dry_run)
    if not args.dry_run:
        print(f"Done ({n} removed). Rebuild with ./build.sh.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
