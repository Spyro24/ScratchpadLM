# opencodeSkill — `readpy.py`

Structural reader for Python files (stdlib only: `ast`). Lets you inspect a
file's shape before reading it in full.

## Usage

```bash
python3 opencodeSkill/readpy.py tree <file>              # class/function tree + line numbers
python3 opencodeSkill/readpy.py treefull <file>          # tree + leading `#` comments + docstrings
python3 opencodeSkill/readpy.py class <file> <Name>      # full class source (methods + code)
python3 opencodeSkill/readpy.py function <file> <name>   # full source of one function
```

`classfull` is accepted as an alias of `class`. Names accept dotted paths
(`Foo.bar`, `outer.inner`). Exit code `0` on success, `1` on error (file
missing, invalid Python, or unknown name — details go to stderr, including
the file's top-level names as a hint).

## Examples

```bash
$ python3 opencodeSkill/readpy.py tree tinylm/sched.py
tinylm/sched.py
class LR_Scheduler @10
  def __init__ @13
  def get_lr @21
...

$ python3 opencodeSkill/readpy.py function tinylm/sched.py LR_Scheduler.get_lr
    def get_lr(self, step: int) -> float:
        ...
```

## Notes

- `treefull` shows contiguous `#` comment lines directly above each
  definition plus its full docstring; it never executes the file.
- Only direct class/function children are listed (definitions nested inside
  `if`/`try` blocks at the same level are skipped).
