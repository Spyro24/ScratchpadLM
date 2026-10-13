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

PALETTE = {
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


def qss_theme() -> str:
    """Qt stylesheet derived from the shared PALETTE."""
    p = PALETTE
    return f"""
    QMainWindow, QDialog {{ background: {p['bg']}; }}
    QWidget {{ font-size: 13px; color: {p['text']}; }}
    QTabWidget::pane {{ border: 1px solid {p['border']}; background: {p['surface']}; top: -1px; }}
    QTabBar::tab {{ padding: 8px 18px; background: #EAECEE; border: 1px solid {p['border']};
                    border-bottom: none; border-top-left-radius: 4px; border-top-right-radius: 4px; }}
    QTabBar::tab:selected {{ background: {p['surface']}; color: {p['accent']}; font-weight: bold; }}
    QTabBar::tab:hover {{ background: #FDFEFE; }}
    QPushButton {{ padding: 5px 12px; border: 1px solid #CCD1D1; border-radius: 4px;
                   background: {p['surface']}; }}
    QPushButton:hover {{ border-color: {p['accent']}; }}
    QPushButton:pressed {{ background: #EAF2F8; }}
    QPushButton:disabled {{ color: #AAB7B8; background: #ECF0F1; border-color: #E5E7E9; }}
    QPushButton#sendButton {{ background: {p['accent']}; color: white; font-weight: bold;
                              border: none; padding: 7px 20px; }}
    QPushButton#sendButton:hover {{ background: #2471A3; }}
    QPushButton#sendButton:disabled {{ background: #AEB6BF; }}
    QLineEdit, QTextEdit, QPlainTextEdit, QSpinBox, QDoubleSpinBox, QComboBox, QListWidget {{
        border: 1px solid #CCD1D1; border-radius: 4px; padding: 4px;
        background: {p['surface']}; selection-background-color: {p['accent']};
    }}
    QGroupBox {{ border: 1px solid {p['border']}; border-radius: 4px; margin-top: 10px;
                 padding: 10px 6px 6px 6px; font-weight: bold; }}
    QGroupBox::title {{ subcontrol-origin: margin; left: 8px; padding: 0 4px; }}
    QProgressBar {{ border: 1px solid #CCD1D1; border-radius: 4px; text-align: center;
                    background: {p['surface']}; min-height: 18px; }}
    QProgressBar::chunk {{ background: {p['accent']}; border-radius: 3px; }}
    QScrollBar:vertical {{ width: 12px; background: #ECF0F1; margin: 0; }}
    QScrollBar::handle:vertical {{ background: #BDC3C7; border-radius: 6px; min-height: 24px; }}
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
    QScrollBar:horizontal {{ height: 12px; background: #ECF0F1; margin: 0; }}
    QScrollBar::handle:horizontal {{ background: #BDC3C7; border-radius: 6px; min-width: 24px; }}
    QCheckBox::indicator, QRadioButton::indicator {{ width: 14px; height: 14px; }}
    QStatusBar {{ background: #EAECEE; border-top: 1px solid {p['border']}; }}
    """


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
    lines = ["# TinyLM Configuration"]
    for section, values in data.items():
        lines.append(f"{section}:")
        for key, value in values.items():
            if value is None:
                rendered = "null"
            elif isinstance(value, bool):
                rendered = "true" if value else "false"
            else:
                rendered = str(value)
            lines.append(f"  {key}: {rendered}")
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
