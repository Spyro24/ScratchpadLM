---
name: opencodeSkill
description: Skills for the agent coding this system. Use readpy to scout a Python file's structure (tree/treefull/class/function) before reading it in full. Triggers: working in this repo, locating a class or function, keeping context small.
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
