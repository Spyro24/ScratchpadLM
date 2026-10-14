"""PySide6 (Qt6) primary GUI for ScratchpadLM: `MainWindow` + `run_qt_app`.

`MainWindow` is composed from the `uiqt/` mixin modules (one per
tab/feature). This is the primary UI toolkit; Tkinter remains available as
a stdlib-only fallback (``uitk``); both share the same colors
(``ui_base.PALETTE``), helpers, and background-thread model.
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
from uiqt.tab_tasks import TasksTabMixin
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
    TasksTabMixin,
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
