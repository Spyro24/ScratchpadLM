---
name: opencodeSkill
description: Skills for the agent coding this system. Use readpy to scout a Python file's structure (tree/treefull/class/function) before reading it in full. Use web to search/fetch web content. Use test_commit to run tests and auto-commit on success. Triggers: working in this repo, locating a class or function, keeping context small, needing web info, testing and committing changes.
---

# opencodeSkill — agent skills for this repo

This folder holds helper skills for the agent that codes this system.
Each skill is a stdlib-only script plus a short doc.

## Skills

### read (`readpy.py`) — Python file-structure reader

Scout a file's shape with few tokens before doing a full read.

```bash
python3 opencodeSkill/readpy.py tree <file>              # class/def tree + line numbers
python3 opencodeSkill/readpy.py treefull <file>          # + leading `#` comments + docstrings
python3 opencodeSkill/readpy.py class <file> <Name>      # full class source (methods + code)
python3 opencodeSkill/readpy.py function <file> <name>   # full source of one function
```

Rules:

- Always run `tree` first on an unfamiliar file; read only the ranges you
  need (`read` with offset/limit), and prefer `function`/`class` over full
  file reads.
- Names accept dotted paths (`Foo.bar`, `outer.inner`); `classfull` is an
  alias of `class`.
- Exit code `0` on success, `1` on error (missing file, invalid Python,
  unknown name — stderr names the file's top-level definitions as a hint).
- See `README.md` for examples and output format.

### web (`web.py`) — stdlib-only web search & fetch

Search the web or fetch page content (text/markdown) with no API key.

```bash
python3 opencodeSkill/web.py search "query" [--limit N]   # web search → JSON list
python3 opencodeSkill/web.py fetch <url>                   # fetch page as plain text
python3 opencodeSkill/web.py fetchmd <url>                 # fetch page as markdown
```

Output:
- `search`: JSON array of `{title, url, snippet}` objects.
- `fetch`/`fetchmd`: raw text/markdown to stdout.

Notes:
- Uses DuckDuckGo HTML endpoint (no API key, stdlib only: `urllib`, `ssl`, `html.parser`).
- `fetchmd` does a best-effort HTML→markdown conversion (headers, links, lists, code, bold/italic).
- Exit code `0` on success, `1` on error (stderr has details).

### test_commit (`test_commit.py`) — automated test & commit

Run the full test suite; on success, stage all changes and commit. On failure, print every failure and abort.

```bash
python3 opencodeSkill/test_commit.py "commit message"           # run tests → commit if all pass
python3 opencodeSkill/test_commit.py --dry-run "msg"            # test only, no commit
python3 opencodeSkill/test_commit.py --filter "pattern" "msg"   # run filtered tests only
```

Behavior:
- Runs `python3 test_all.py` (with optional filter) — must pass (exit code 0).
- On success: `git add -A` then `git commit -m "message"`.
- On failure: prints each failure line, shows tail of test output, exits non-zero — **no commit**.
- `--dry-run` stops after tests (useful for CI/validation).
- Stdlib only (`subprocess`, `sys`).

Exit codes: `0` = tests passed (+ committed unless --dry-run), `1` = tests failed or commit error.