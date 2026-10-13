"""
Shared UI layer for ScratchpadLM — toolkit detection, theming, helpers.

Toolkits:
- **PySide6 (Qt6)** is the primary toolkit (``ui_qt.py``). It is an external
  dependency, so it is imported lazily and never required.
- **Tkinter** is the stdlib-only fallback (``ui.py``) — the project stays
  usable with zero external dependencies.

``detect_toolkit()`` picks the toolkit at startup (PySide6 preferred,
Tkinter fallback). Set ``SCRATCHPADLM_UI=tkinter`` or ``SCRATCHPADLM_UI=pyside6``
to force a specific toolkit.

This module must not import any GUI toolkit at module level.
"""

import os
import queue
import sys
import threading
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple

from config import Config

# ---------------------------------------------------------------------------
# Paths & config (shared by both toolkits)
# ---------------------------------------------------------------------------

MODEL_PATH = "tinyLM"
TOKENIZER_PATH = "bpe_merges.txt"
CONFIG_PATH = "config.yaml"

# ---------------------------------------------------------------------------
# Theme palette — shared so both toolkits look consistent
# ---------------------------------------------------------------------------

# Light theme (default)
PALETTE_LIGHT = {
    "accent": "#2E86C1",    # user / training / primary actions
    "success": "#27AE60",   # assistant / generation
    "muted": "#7F8C8D",     # system messages
    "error": "#E74C3C",     # errors
    "warning": "#F39C12",   # warnings / middling ratings
    "bg": "#F4F6F7",        # window background
    "surface": "#FFFFFF",   # card / panel background
    "text": "#2C3E50",      # primary text
    "border": "#D5DBDB",    # borders
}

# Dark theme - improved contrast for readability
PALETTE_DARK = {
    "accent": "#85C1E9",      # brighter blue for buttons/links
    "success": "#82E0AA",     # brighter green for assistant
    "muted": "#AAB7B8",       # lighter gray for system/muted text
    "error": "#F1948A",       # softer red for errors
    "warning": "#F7DC6F",     # brighter yellow for warnings
    "bg": "#1A1A2E",          # very dark navy (not pure black)
    "surface": "#252A3E",     # dark card background
    "text": "#F0F0F0",        # near-white primary text
    "border": "#3D445A",      # subtle border color
}

# Default palette (light)
PALETTE = PALETTE_LIGHT


def get_palette(theme: str = "light") -> dict:
    """Get palette for theme: 'light', 'dark', or 'system'."""
    if theme == "dark":
        return PALETTE_DARK
    if theme == "light":
        return PALETTE_LIGHT
    # system - detect OS preference
    return _detect_system_palette()


def _detect_system_palette() -> dict:
    """Detect system color scheme preference."""
    try:
        # Try Qt first (if available)
        from PySide6.QtWidgets import QApplication
        app = QApplication.instance()
        if app is not None:
            from PySide6.QtGui import QGuiApplication
            style_hints = QGuiApplication.styleHints()
            if hasattr(style_hints, "colorScheme"):
                scheme = style_hints.colorScheme()
                from PySide6.QtGui import QColorScheme
                if scheme == QColorScheme.Dark:
                    return PALETTE_DARK
    except Exception:
        pass
    
    # Fallback: Linux gsettings / XDG heuristics
    import subprocess, os
    try:
        # GNOME
        result = subprocess.run(
            ["gsettings", "get", "org.gnome.desktop.interface", "color-scheme"],
            capture_output=True, text=True, timeout=1
        )
        if "dark" in result.stdout.lower():
            return PALETTE_DARK
    except Exception:
        pass
    
    try:
        # KDE / generic XDG
        desktop = os.environ.get("XDG_CURRENT_DESKTOP", "").lower()
        if "kde" in desktop:
            result = subprocess.run(
                ["kreadconfig5", "--group", "General", "--key", "ColorScheme"],
                capture_output=True, text=True, timeout=1
            )
            if "breeze" in result.stdout.lower() and "dark" in result.stdout.lower():
                return PALETTE_DARK
    except Exception:
        pass
    
    return PALETTE_LIGHT


def qss_theme(theme: str = "light") -> str:
    """Qt stylesheet derived from the shared palette for given theme."""
    p = get_palette(theme)
    return f"""
    QMainWindow, QDialog {{ background: {p['bg']}; }}
    QWidget {{ font-size: 13px; color: {p['text']}; }}
    QTabWidget::pane {{ border: 1px solid {p['border']}; background: {p['surface']}; top: -1px; }}
    QTabBar::tab {{ padding: 8px 18px; background: {p['bg']}; border: 1px solid {p['border']};
                    border-bottom: none; border-top-left-radius: 4px; border-top-right-radius: 4px; color: {p['text']}; }}
    QTabBar::tab:selected {{ background: {p['surface']}; color: {p['accent']}; font-weight: bold; }}
    QTabBar::tab:hover {{ background: {p['border']}; }}
    QPushButton {{ padding: 5px 12px; border: 1px solid {p['border']}; border-radius: 4px;
                   background: {p['surface']}; color: {p['text']}; }}
    QPushButton:hover {{ border-color: {p['accent']}; }}
    QPushButton:pressed {{ background: {p['bg']}; }}
    QPushButton:disabled {{ color: {p['muted']}; background: {p['bg']}; border-color: {p['border']}; }}
    QPushButton#sendButton {{ background: {p['accent']}; color: white; font-weight: bold;
                              border: none; padding: 7px 20px; }}
    QPushButton#sendButton:hover {{ background: {p['accent']}; opacity: 0.85; }}
    QPushButton#sendButton:disabled {{ background: {p['muted']}; }}
    QLineEdit, QTextEdit, QPlainTextEdit, QSpinBox, QDoubleSpinBox, QComboBox, QListWidget {{
        border: 1px solid {p['border']}; border-radius: 4px; padding: 4px;
        background: {p['surface']}; color: {p['text']}; selection-background-color: {p['accent']};
    }}
    QGroupBox {{ border: 1px solid {p['border']}; border-radius: 4px; margin-top: 10px;
                 padding: 10px 6px 6px 6px; font-weight: bold; color: {p['text']}; }}
    QGroupBox::title {{ subcontrol-origin: margin; left: 8px; padding: 0 4px; }}
    QProgressBar {{ border: 1px solid {p['border']}; border-radius: 4px; text-align: center;
                    background: {p['surface']}; min-height: 18px; color: {p['text']}; }}
    QProgressBar::chunk {{ background: {p['accent']}; border-radius: 3px; }}
    QScrollArea {{ background: {p['bg']}; border: none; }}
    QScrollArea > QWidget > QWidget {{ background: {p['bg']}; }}
    QScrollArea QWidget {{ background: {p['bg']}; }}
    QAbstractScrollArea {{ background: {p['bg']}; }}
    QAbstractScrollArea > QWidget {{ background: {p['bg']}; }}
    QScrollBar:vertical {{ width: 12px; background: {p['bg']}; margin: 0; }}
    QScrollBar::handle:vertical {{ background: {p['muted']}; border-radius: 6px; min-height: 24px; }}
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
    QScrollBar:horizontal {{ height: 12px; background: {p['bg']}; margin: 0; }}
    QScrollBar::handle:horizontal {{ background: {p['muted']}; border-radius: 6px; min-width: 24px; }}
    QCheckBox::indicator, QRadioButton::indicator {{ width: 14px; height: 14px; }}
    QStatusBar {{ background: {p['bg']}; border-top: 1px solid {p['border']}; color: {p['text']}; }}
    QToolTip {{ background: {p['surface']}; color: {p['text']}; border: 1px solid {p['border']}; }}
    """


def apply_tk_theme(root, theme: str = "light") -> dict:
    """Apply theme to Tkinter root window and return palette dict."""
    p = get_palette(theme)
    root.configure(bg=p["bg"])
    # Style ttk widgets if available
    try:
        import tkinter.ttk as ttk
        style = ttk.Style()
        style.theme_use("clam")
        style.configure(".", background=p["bg"], foreground=p["text"], fieldbackground=p["surface"])
        style.configure("TFrame", background=p["bg"])
        style.configure("TLabel", background=p["bg"], foreground=p["text"])
        style.configure("TButton", background=p["surface"], foreground=p["text"], bordercolor=p["border"])
        style.map("TButton", background=[("active", p["accent"]), ("pressed", p["bg"])],
                  foreground=[("active", "white")])
        style.configure("TEntry", fieldbackground=p["surface"], foreground=p["text"], bordercolor=p["border"])
        style.configure("TCombobox", fieldbackground=p["surface"], foreground=p["text"], bordercolor=p["border"])
        style.configure("TSpinbox", fieldbackground=p["surface"], foreground=p["text"], bordercolor=p["border"])
        style.configure("TNotebook", background=p["bg"], borderwidth=0)
        style.configure("TNotebook.Tab", background=p["bg"], foreground=p["text"], padding=[12, 8],
                        bordercolor=p["border"])
        style.map("TNotebook.Tab", background=[("selected", p["surface"])], foreground=[("selected", p["accent"])])
        style.configure("TLabelframe", background=p["bg"], foreground=p["text"], bordercolor=p["border"])
        style.configure("TLabelframe.Label", background=p["bg"], foreground=p["text"])
        style.configure("TProgressbar", background=p["accent"], troughcolor=p["surface"],
                        bordercolor=p["border"], lightcolor=p["accent"], darkcolor=p["accent"])
        style.configure("Horizontal.TProgressbar", background=p["accent"])
        style.configure("Vertical.TProgressbar", background=p["accent"])
        style.configure("TScrollbar", background=p["bg"], troughcolor=p["bg"],
                        bordercolor=p["bg"], arrowcolor=p["text"])
        style.configure("TCheckbutton", background=p["bg"], foreground=p["text"])
        style.configure("TRadiobutton", background=p["bg"], foreground=p["text"])
    except Exception:
        pass
    return p


# ---------------------------------------------------------------------------
# Toolkit detection & launcher
# ---------------------------------------------------------------------------

def has_pyside6() -> bool:
    """True when PySide6 (Qt6) is importable."""
    try:
        import PySide6  # noqa: F401
        return True
    except Exception:
        return False


def qt_platform_check() -> Tuple[bool, str]:
    """Preflight Qt GUI availability WITHOUT constructing a QApplication.

    When Qt cannot initialize a platform plugin (e.g. Qt >= 6.5 requires
    libxcb-cursor / the xcb-cursor0 package on X11) it calls ``abort()`` —
    the process dies with a core dump and no Python exception is raised.
    Checking the platform first lets :func:`run_gui` fall back to Tkinter
    instead of crashing.

    Returns ``(ok, reason)``.
    """
    if not has_pyside6():
        return False, "PySide6 is not installed"
    if os.environ.get("QT_QPA_PLATFORM", "").strip():
        # Explicit platform requested (offscreen, minimal, wayland, ...).
        # Trust it — these plugins have no libxcb-cursor dependency.
        return True, ""
    display = os.environ.get("DISPLAY", "").strip()
    wayland = os.environ.get("WAYLAND_DISPLAY", "").strip()
    if not display and not wayland:
        return False, "no DISPLAY/WAYLAND_DISPLAY (headless session)"
    if display and not wayland:
        # Qt will select the xcb plugin; Qt >= 6.5 hard-requires
        # libxcb-cursor there. Check it before it can abort the process.
        import ctypes.util

        if ctypes.util.find_library("xcb-cursor") is None:
            return False, (
                "Qt xcb platform needs libxcb-cursor "
                "(install the xcb-cursor0 / libxcb-cursor0 package)"
            )
    return True, ""


def detect_toolkit() -> str:
    """Return ``"pyside6"`` or ``"tkinter"`` (PySide6 preferred).

    Honors the ``SCRATCHPADLM_UI`` environment variable override.
    """
    choice = os.environ.get("SCRATCHPADLM_UI", "").strip().lower()
    if choice in ("pyside6", "qt", "qt6"):
        return "pyside6" if has_pyside6() else "tkinter"
    if choice in ("tkinter", "tk"):
        return "tkinter"
    return "pyside6" if has_pyside6() else "tkinter"


def run_gui() -> Any:
    """Launch the GUI: PySide6 primary, Tkinter stdlib fallback.

    Qt platform plugins are preflight-checked (see :func:`qt_platform_check`)
    because a failing plugin would abort the process instead of raising.
    Set ``SCRATCHPADLM_UI=pyside6`` to skip the fallback and get a clean
    error instead.

    Returns the toolkit that was actually used.
    """
    forced_pyside6 = os.environ.get("SCRATCHPADLM_UI", "").strip().lower() in (
        "pyside6", "qt", "qt6",
    )
    if detect_toolkit() == "pyside6":
        ok, reason = qt_platform_check()
        if not ok:
            if forced_pyside6:
                raise RuntimeError(f"PySide6 requested but unavailable: {reason}")
            sys.stderr.write(
                f"[ui] PySide6 unavailable ({reason}); falling back to Tkinter\n"
            )
        else:
            try:
                from ui_qt import run_qt_app
                run_qt_app()
                return "pyside6"
            except (Exception, SystemExit) as exc:  # plugin init, broken install, ...
                if forced_pyside6:
                    raise
                sys.stderr.write(
                    f"[ui] PySide6 failed ({exc}); falling back to Tkinter\n"
                )

    import tkinter as tk

    App = getattr(sys.modules.get("__main__"), "App", None)
    if App is None:
        from ui import App  # noqa: PLC0415 — lazy: keeps this module toolkit-free
    root = tk.Tk()
    App(root)
    root.mainloop()
    return "tkinter"


# ---------------------------------------------------------------------------
# Background worker (queue-based; toolkit-agnostic)
# ---------------------------------------------------------------------------

class BackgroundWorker:
    """Runs jobs on daemon threads; results are polled by the UI's own timer.

    The UI never touches widgets from worker threads — workers only
    ``post()`` messages, and the UI drains them on its main thread
    (``root.after`` for Tk, ``QTimer`` for Qt).
    """

    def __init__(self):
        self.queue: "queue.Queue[Tuple[str, str, Any]]" = queue.Queue()
        self.busy = False

    def submit(self, label: str, func: Callable, *args) -> None:
        """Run ``func(*args)`` in a daemon thread; deliver result/error."""
        self.busy = True

        def worker():
            try:
                result = func(*args)
                self.queue.put(("ok", label, result))
            except Exception as exc:
                self.queue.put(("error", label, exc))

        threading.Thread(target=worker, daemon=True).start()

    def post(self, kind: str, label: str, payload: Any) -> None:
        """Thread-safe: enqueue a progress/log/stream message."""
        self.queue.put((kind, label, payload))

    def drain(self) -> List[Tuple[str, str, Any]]:
        """Non-blocking: return every pending message (main thread)."""
        out: List[Tuple[str, str, Any]] = []
        while True:
            try:
                out.append(self.queue.get_nowait())
            except queue.Empty:
                break
        return out


# ---------------------------------------------------------------------------
# Shared helpers (no GUI dependencies)
# ---------------------------------------------------------------------------

def load_config(path: str = CONFIG_PATH) -> Config:
    """Load config.yaml, falling back to defaults on any error."""
    try:
        return Config.from_file(path)
    except Exception:
        return Config.from_dict({})


def write_config_yaml(path: str, data: dict) -> None:
    """Write a 2-level config dict as YAML (matches config.py's parser)."""
    def format_value(v, indent=0):
        """Format a value for YAML output."""
        prefix = "  " * indent
        if v is None:
            return "null"
        elif isinstance(v, bool):
            return "true" if v else "false"
        elif isinstance(v, list):
            if not v:
                return "[]"  # Empty list
            lines = []
            for item in v:
                lines.append(f"{prefix}- {format_value(item)}")
            return "\n".join(lines)
        else:
            return str(v)
    
    lines = ["# TinyLM Configuration"]
    for section, values in data.items():
        lines.append(f"{section}:")
        for key, value in values.items():
            formatted = format_value(value, 1)
            if "\n" in formatted:
                # Multi-line (list)
                lines.append(f"  {key}:")
                for line in formatted.split("\n"):
                    lines.append(f"    {line.strip()}")
            else:
                lines.append(f"  {key}: {formatted}")
        lines.append("")
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def parse_bool(raw: str) -> bool:
    """Parse a Settings entry as a boolean (for early_stopping etc.)."""
    s = str(raw).strip().lower()
    if s in ("true", "1", "yes", "on"):
        return True
    if s in ("false", "0", "no", "off"):
        return False
    raise ValueError(f"expected boolean, got '{raw}'")


# cast.__name__ is shown in Settings validation error messages
parse_bool.__name__ = "bool"


def get_available_models() -> List[str]:
    """Available model names from registry, catalog, and local files."""
    from registry import MODEL_REGISTRY  # local: avoids import cycles
    from model_mgmt import MODEL_CATALOG

    models = list(MODEL_REGISTRY.list())
    try:
        for spec in MODEL_CATALOG.list():
            if spec.name not in models and Path(spec.path).exists():
                models.append(spec.name)
    except Exception:
        pass
    if "TinyLM" not in models:
        models.insert(0, "TinyLM")
    return models
