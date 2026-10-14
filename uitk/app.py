"""Tkinter (stdlib fallback) GUI for ScratchpadLM: the `App` class.

`App` is composed from the `uitk/` mixin modules (one per tab/feature).
Launched through ``ui_base.run_gui()`` (or ``main.py``), which auto-detects
the best toolkit at startup and falls back to Tkinter when Qt is
unavailable. Force a toolkit with the ``SCRATCHPADLM_UI`` environment
variable (``pyside6`` or ``tkinter``).
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
from uitk.tab_tasks import TasksTabMixin
from uitk.tab_training import TrainingTabMixin
from uitk.worker import WorkerMixin


class App(
    CoreMixin,
    ShellMixin,
    ChatTabMixin,
    TrainingTabMixin,
    InfoTabMixin,
    AdminTabMixin,
    SettingsTabMixin,
    TasksTabMixin,
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
