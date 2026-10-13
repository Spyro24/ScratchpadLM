"""Read skill for Python files (opencodeSkill).

Structural reader for Python source (stdlib only: `ast`). Subcommands:

    tree FILE              - class/function tree with line numbers
    treefull FILE          - tree plus leading `#` comments and docstrings
    class FILE NAME        - full source of a class (incl. methods and code)
    function FILE NAME     - full source of a single function

NAME accepts dotted paths (e.g. `Foo.bar`, `outer.inner`).
Exit code 0 on success, 1 on error (message goes to stderr).
"""
import argparse
import ast
import sys

DEF_TYPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)


def read_source(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def parse_file(path):
    """Return (source_text, module_node). Raises OSError / SyntaxError."""
    src = read_source(path)
    try:
        tree = ast.parse(src, filename=path)
    except SyntaxError as e:
        raise SyntaxError(f"{path}: invalid Python ({e})") from None
    return src, tree


def _children(node):
    """Direct class/function children of a module, class, or function."""
    return [c for c in ast.iter_child_nodes(node) if isinstance(c, DEF_TYPES)]


def def_start(node):
    """First line belonging to a def (covers decorators above it)."""
    first = node.lineno
    for d in getattr(node, "decorator_list", []):
        if d.lineno < first:
            first = d.lineno
    return first


def leading_comments(lines, start):
    """Contiguous `#` comment lines directly above `start` (1-based)."""
    out = []
    i = start - 2
    while i >= 0 and lines[i].strip().startswith("#"):
        out.append(lines[i].strip())
        i -= 1
    out.reverse()
    return out


def _fmt_doc(doc, pad):
    """Render a docstring as indented quoted lines."""
    mark = '"' * 3
    dl = doc.splitlines() or [""]
    if len(dl) == 1:
        return [pad + mark + dl[0] + mark]
    return [pad + mark + dl[0]] + [pad + d for d in dl[1:]] + [pad + mark]


def _render_tree(lines, node, depth, out, full):
    for child in _children(node):
        kind = "class" if isinstance(child, ast.ClassDef) else "def"
        out.append("  " * depth + f"{kind} {child.name} @{child.lineno}")
        if full:
            pad = "  " * (depth + 1)
            for c in leading_comments(lines, def_start(child)):
                out.append(pad + c)
            doc = ast.get_docstring(child)
            if doc:
                out.extend(_fmt_doc(doc, pad))
        _render_tree(lines, child, depth + 1, out, full)


def tree_text(path):
    """`class`/`def` tree of the file, with line numbers."""
    src, tree = parse_file(path)
    out = [path]
    _render_tree(src.splitlines(), tree, 0, out, full=False)
    return "\n".join(out) + "\n"


def treefull_text(path):
    """Tree plus leading `#` comments and docstrings for each entry."""
    src, tree = parse_file(path)
    out = [path]
    _render_tree(src.splitlines(), tree, 0, out, full=True)
    return "\n".join(out) + "\n"


def find_node(tree, dotted):
    """Resolve a dotted `Class.method` / `outer.inner` path. None if missing."""
    node = tree
    for part in dotted.split("."):
        nxt = None
        for child in ast.iter_child_nodes(node):
            if isinstance(child, DEF_TYPES) and child.name == part:
                nxt = child
                break
        if nxt is None:
            return None
        node = nxt
    return node


def top_names(tree):
    return [c.name for c in _children(tree)]


def class_source(path, name):
    """Full source of class NAME (methods and code included)."""
    src, tree = parse_file(path)
    node = find_node(tree, name)
    if node is None or not isinstance(node, ast.ClassDef):
        raise LookupError(
            f"{path}: class {name!r} not found "
            f"(top level: {', '.join(top_names(tree)) or '-'})")
    seg = ast.get_source_segment(src, node)
    if seg is None:
        raise LookupError(f"{path}: could not extract source for {name!r}")
    return seg + "\n"


def function_source(path, name):
    """Full source of function NAME (dotted paths allowed)."""
    src, tree = parse_file(path)
    node = find_node(tree, name)
    if node is None or not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        raise LookupError(
            f"{path}: function {name!r} not found "
            f"(top level: {', '.join(top_names(tree)) or '-'})")
    seg = ast.get_source_segment(src, node)
    if seg is None:
        raise LookupError(f"{path}: could not extract source for {name!r}")
    return seg + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="readpy.py", description="Structural reader for Python files.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("tree", help="list class/function tree with line numbers")
    p.add_argument("file")
    p = sub.add_parser("treefull", help="tree plus comments and docstrings")
    p.add_argument("file")
    p = sub.add_parser("class", aliases=["classfull"],
                       help="full source of a class (incl. methods)")
    p.add_argument("file")
    p.add_argument("name")
    p = sub.add_parser("function", help="full source of a single function")
    p.add_argument("file")
    p.add_argument("name")
    args = ap.parse_args(argv)
    try:
        if args.cmd == "tree":
            sys.stdout.write(tree_text(args.file))
        elif args.cmd == "treefull":
            sys.stdout.write(treefull_text(args.file))
        elif args.cmd in ("class", "classfull"):
            sys.stdout.write(class_source(args.file, args.name))
        elif args.cmd == "function":
            sys.stdout.write(function_source(args.file, args.name))
    except (OSError, SyntaxError, LookupError) as e:
        print(f"readpy: error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
