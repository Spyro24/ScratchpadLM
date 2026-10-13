"""
PySide6 (Qt6) primary GUI for ScratchpadLM.

This is the primary UI toolkit (roadmap 0a: "UI Toolkit Migration"). Tkinter
remains available as a stdlib-only fallback (``ui.py``); both toolkits share
the same colors (``ui_base.PALETTE``), helpers, and background-thread model.

Tabs: **Chat**, **Training**, **Model Info**, **Admin**, **Settings**, **Logs**.

Threading rules (same as the Tkinter app):
- Qt widgets are touched only on the main thread.
- Jobs run in daemon threads via ``ui_base.BackgroundWorker``.
- Worker results are delivered by posting on the queue; a ``QTimer`` drains
  the queue on the main thread every 100 ms.

Implementation lives in the ``uiqt/`` package (one mixin module per tab /
feature, split from the former monolithic ``ui_qt.py``). This module keeps
the public entry points: ``MainWindow`` and ``run_qt_app``.
"""

from PySide6.QtWidgets import QApplication, QMainWindow

from ui_base import CONFIG_PATH, qss_theme
from uiqt.act_admin import AdminActionsMixin
from uiqt.act_settings import SettingsActionsMixin
from uiqt.act_training import TrainingActionsMixin
from uiqt.chat import ChatMixin
from uiqt.core import CoreMixin
from uiqt.logging import LoggingMixin
from uiqt.result import ResultMixin
from uiqt.shell import ShellMixin
from uiqt.tab_admin import AdminTabMixin
from uiqt.tab_chat import ChatTabMixin
from uiqt.tab_info import InfoTabMixin
from uiqt.tab_logs import LogsTabMixin
from uiqt.tab_settings import SettingsTabMixin
from uiqt.tab_training import TrainingTabMixin
from uiqt.worker import WorkerMixin


class MainWindow(
    CoreMixin,
    ShellMixin,
    ChatTabMixin,
    TrainingTabMixin,
    InfoTabMixin,
    AdminTabMixin,
    SettingsTabMixin,
    LogsTabMixin,
    ChatMixin,
    WorkerMixin,
    ResultMixin,
    LoggingMixin,
    TrainingActionsMixin,
    AdminActionsMixin,
    SettingsActionsMixin,
    QMainWindow,
):
    """ScratchpadLM main window (PySide6)."""


def run_qt_app() -> int:
    """Launch the PySide6 app (called by ui_base.run_gui)."""
    import sys

    # Load config early to get theme
    from config import Config
    try:
        cfg = Config.from_file(CONFIG_PATH)
        theme = cfg.get("ui", "theme", "dark")
    except Exception:
        theme = "dark"

    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("ScratchpadLM")
    app.setStyleSheet(qss_theme(theme))
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(run_qt_app())
