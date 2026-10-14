"""PySide6 UI mixin. Part of the `uiqt/` package split from `ui_qt.py`."""
from PySide6.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QPushButton,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
from PySide6.QtCore import Qt

from scratchpad import TaskPad, Scratchpad
from uiqt.widgets import MONO


class TasksTabMixin:
    def _build_tasks_tab(self) -> QWidget:
        root = QWidget()
        v = QVBoxLayout(root)
        v.setContentsMargins(4, 4, 4, 4)

        top = QHBoxLayout()
        top.addWidget(QLabel("TaskPad tasks + Scratchpad entries (read from disk)."))
        top.addStretch(1)
        self.tasks_refresh_btn = QPushButton("Refresh")
        self.tasks_refresh_btn.clicked.connect(self.refresh_tasks_tab)
        top.addWidget(self.tasks_refresh_btn)
        v.addLayout(top)

        splitter = QSplitter(Qt.Orientation.Vertical)
        v.addWidget(splitter, 1)

        # TaskPad panel
        task_group = QGroupBox("TaskPad")
        tg = QVBoxLayout(task_group)
        self.task_list = QListWidget()
        self.task_list.currentRowChanged.connect(self._on_task_selected)
        tg.addWidget(self.task_list, 1)
        self.task_detail = QTextEdit()
        self.task_detail.setReadOnly(True)
        self.task_detail.setFont(MONO)
        self.task_detail.setMaximumHeight(160)
        tg.addWidget(self.task_detail)
        splitter.addWidget(task_group)

        # Scratchpad panel
        sp_group = QGroupBox("Scratchpad")
        sg = QVBoxLayout(sp_group)
        self.scratch_list = QListWidget()
        self.scratch_list.currentRowChanged.connect(self._on_scratch_selected)
        sg.addWidget(self.scratch_list, 1)
        self.scratch_detail = QTextEdit()
        self.scratch_detail.setReadOnly(True)
        self.scratch_detail.setFont(MONO)
        self.scratch_detail.setMaximumHeight(160)
        sg.addWidget(self.scratch_detail)
        splitter.addWidget(sp_group)

        self._tasks_cache = []
        self._scratch_cache = []
        self.refresh_tasks_tab()
        return root

    def refresh_tasks_tab(self):
        """Reload TaskPad tasks + Scratchpad entries from disk."""
        try:
            self._tasks_cache = TaskPad().list_tasks()
        except Exception as exc:
            self._tasks_cache = []
            self.task_detail.setPlainText(f"Failed to load TaskPad: {exc}")
        self.task_list.clear()
        for t in self._tasks_cache:
            status = t.status.value if hasattr(t.status, "value") else str(t.status)
            ttype = t.type.value if hasattr(t.type, "value") else str(t.type)
            prompt = (t.prompt or "")[:80].replace("\n", " ")
            self.task_list.addItem(f"[{status}] {ttype} {t.id}: {prompt}")
        if self._tasks_cache:
            self.task_list.setCurrentRow(0)
        else:
            self.task_detail.setPlainText("No tasks.")

        try:
            self._scratch_cache = Scratchpad().get_recent(200)
        except Exception as exc:
            self._scratch_cache = []
            self.scratch_detail.setPlainText(f"Failed to load Scratchpad: {exc}")
        self.scratch_list.clear()
        for e in self._scratch_cache:
            thought = (e.thought or "")[:80].replace("\n", " ")
            self.scratch_list.addItem(f"[{e.model}] step {e.step}: {thought}")
        if self._scratch_cache:
            self.scratch_list.setCurrentRow(len(self._scratch_cache) - 1)
        else:
            self.scratch_detail.setPlainText("No entries.")

    def _on_task_selected(self, row: int):
        if 0 <= row < len(self._tasks_cache):
            t = self._tasks_cache[row]
            lines = [
                f"id: {t.id}",
                f"type: {t.type.value if hasattr(t.type, 'value') else t.type}",
                f"status: {t.status.value if hasattr(t.status, 'value') else t.status}",
                f"model: {t.model_name}",
                f"prompt: {t.prompt}",
                f"result: {t.result or '—'}",
                f"error: {t.error or '—'}",
                f"dependencies: {', '.join(t.dependencies) if t.dependencies else '—'}",
            ]
            self.task_detail.setPlainText("\n".join(lines))

    def _on_scratch_selected(self, row: int):
        if 0 <= row < len(self._scratch_cache):
            e = self._scratch_cache[row]
            self.scratch_detail.setPlainText(
                f"[{e.model}] step {e.step}\nthought: {e.thought}\nresult: {e.result or '—'}"
            )
