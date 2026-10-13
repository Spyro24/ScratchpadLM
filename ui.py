"""Tkinter (stdlib fallback) GUI for ScratchpadLM.

PySide6 (Qt6) is the primary toolkit — see ``ui_qt.py``. Both toolkits are
launched through ``ui_base.run_gui()``, which auto-detects the best toolkit
at startup and falls back to Tkinter when Qt is unavailable, so the project
remains usable with the standard library alone. Force a toolkit with the
``SCRATCHPADLM_UI`` environment variable (``pyside6`` or ``tkinter``).

Implementation lives in the ``uitk/`` package (one mixin module per tab /
feature, split from the former monolithic ``ui.py``). This module keeps the
public entry points: ``App`` and ``_write_config_yaml``.
"""

from uitk.act_admin import AdminActionsMixin
from uitk.act_chat import ChatActionsMixin
from uitk.act_settings import SettingsActionsMixin
from uitk.act_training import TrainingActionsMixin
from uitk.core import CoreMixin
from uitk.history import HistoryMixin
from uitk.info import InfoMixin
from uitk.result import ResultMixin
from uitk.shell import ShellMixin
from uitk.tab_admin import AdminTabMixin
from uitk.tab_chat import ChatTabMixin
from uitk.tab_info import InfoTabMixin
from uitk.tab_logs import LogsTabMixin
from uitk.tab_settings import SettingsTabMixin
from uitk.tab_training import TrainingTabMixin
from uitk.worker import WorkerMixin
from ui_base import write_config_yaml as _write_config_yaml


class App(
    CoreMixin,
    ShellMixin,
    ChatTabMixin,
    TrainingTabMixin,
    InfoTabMixin,
    AdminTabMixin,
    SettingsTabMixin,
    LogsTabMixin,
    HistoryMixin,
    WorkerMixin,
    TrainingActionsMixin,
    ChatActionsMixin,
    AdminActionsMixin,
    SettingsActionsMixin,
    ResultMixin,
    InfoMixin,
):
    """Tkinter (stdlib fallback) GUI for ScratchpadLM."""


if __name__ == "__main__":
    # PySide6 (Qt6) is preferred; falls back to this Tkinter app when Qt
    # is unavailable. Force a toolkit: SCRATCHPADLM_UI=pyside6|tkinter
    from ui_base import run_gui
    run_gui()
