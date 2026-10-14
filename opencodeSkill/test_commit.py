#!/usr/bin/env python3
"""
test_commit.py — Automated test & commit skill.

Runs the full test suite. If all tests pass, stages all changes and commits
with the provided message. If any test fails, prints every failure and aborts.

Usage:
  python3 opencodeSkill/test_commit.py "commit message"
  python3 opencodeSkill/test_commit.py --dry-run "commit message"  # test only, no commit
  python3 opencodeSkill/test_commit.py --filter "tokenizer" "commit message"  # run filtered tests
"""

import sys
import subprocess
import re
from typing import List, Tuple, Optional


def run_tests(filter_pattern: Optional[str] = None) -> Tuple[bool, List[str], str]:
    """
    Run test_all.py with optional filter.
    Returns (success, failures_list, full_output).
    """
    cmd = [sys.executable, "test_all.py"]
    if filter_pattern:
        cmd.append(filter_pattern)

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,  # 5 min timeout
            cwd="/home/spyro/s24ai"
        )
        output = result.stdout + result.stderr
        success = result.returncode == 0

        failures = []
        if not success:
            # Parse failures from output
            # Test output format: "FAIL  test_name" or similar
            for line in output.splitlines():
                line = line.strip()
                if line.startswith("FAIL") or "FAILED" in line or "ERROR" in line:
                    failures.append(line)
                # Also capture traceback lines if present
                if "Traceback (most recent call last)" in line:
                    failures.append(line)

            # If no explicit FAIL lines but non-zero exit, include last ~20 lines
            if not failures:
                failures = output.strip().splitlines()[-20:]

        return success, failures, output

    except subprocess.TimeoutExpired:
        return False, ["Test suite timed out after 300 seconds"], ""
    except Exception as e:
        return False, [f"Failed to run tests: {e}"], ""


def git_status() -> str:
    """Get git status output."""
    try:
        result = subprocess.run(
            ["git", "status", "--short"],
            capture_output=True,
            text=True,
            cwd="/home/spyro/s24ai"
        )
        return result.stdout.strip()
    except Exception:
        return ""


def git_add_all() -> bool:
    """Stage all changes."""
    try:
        subprocess.run(
            ["git", "add", "-A"],
            check=True,
            cwd="/home/spyro/s24ai"
        )
        return True
    except subprocess.CalledProcessError:
        return False


def git_commit(message: str) -> Tuple[bool, str]:
    """Commit with message. Returns (success, output)."""
    try:
        result = subprocess.run(
            ["git", "commit", "-m", message],
            capture_output=True,
            text=True,
            cwd="/home/spyro/s24ai"
        )
        return result.returncode == 0, result.stdout + result.stderr
    except Exception as e:
        return False, str(e)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    dry_run = False
    filter_pattern = None
    message = None

    # Parse args
    args = sys.argv[1:]
    if args[0] == "--dry-run":
        dry_run = True
        args = args[1:]
    if args and args[0] == "--filter":
        if len(args) < 2:
            print("Error: --filter requires a pattern")
            sys.exit(1)
        filter_pattern = args[1]
        args = args[2:]
    if args:
        message = " ".join(args)

    if not message:
        print("Error: commit message required")
        print(__doc__)
        sys.exit(1)

    # Show status
    status = git_status()
    if status:
        print("Git status:")
        print(status)
        print()

    # Run tests
    print(f"Running tests{' with filter: ' + filter_pattern if filter_pattern else ''}...")
    success, failures, output = run_tests(filter_pattern)

    if success:
        print("✓ All tests passed")
        if dry_run:
            print("(dry-run: skipping commit)")
            sys.exit(0)

        # Stage and commit
        print("Staging changes...")
        if not git_add_all():
            print("✗ Failed to stage changes")
            sys.exit(1)

        print(f"Committing: {message}")
        commit_ok, commit_out = git_commit(message)
        if commit_ok:
            print("✓ Commit successful")
            if commit_out.strip():
                print(commit_out)
            sys.exit(0)
        else:
            print("✗ Commit failed:")
            print(commit_out)
            sys.exit(1)
    else:
        print(f"✗ Tests failed ({len(failures)} failure(s) reported):")
        print()
        for i, failure in enumerate(failures, 1):
            print(f"  {i}. {failure}")
        print()
        print("Full test output:")
        print("-" * 60)
        print(output[-3000:] if len(output) > 3000 else output)  # Last 3000 chars
        print("-" * 60)
        print("\nAborting commit.")
        sys.exit(1)


if __name__ == "__main__":
    main()