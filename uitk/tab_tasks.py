"""Tkinter UI mixin (stdlib fallback). Part of the `uitk/` package split from `ui.py`."""
import tkinter as tk
from tkinter import scrolledtext

from scratchpad import TaskPad, Scratchpad


class TasksTabMixin:
    def _build_tasks_tab(self, tab):
        """Tasks tab: TaskPad tasks + Scratchpad entries (read from disk)."""
        tab.rowconfigure(1, weight=1)
        tab.rowconfigure(2, weight=1)
        tab.columnconfigure(0, weight=1)

        top = tk.Frame(tab)
        top.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        tk.Label(top, text="TaskPad tasks + Scratchpad entries (read from disk).",
                 font=("Consolas", 9)).pack(side="left")
        tk.Button(top, text="Refresh", command=self.refresh_tasks_tab, width=12).pack(side="right")

        task_frame = tk.LabelFrame(tab, text="TaskPad", padx=10, pady=10)
        task_frame.grid(row=1, column=0, sticky="nsew", pady=(0, 8))
        task_frame.rowconfigure(0, weight=1)
        task_frame.columnconfigure(0, weight=1)
        task_frame.columnconfigure(1, weight=2)
        self.task_listbox = tk.Listbox(task_frame, font=("Consolas", 9))
        self.task_listbox.grid(row=0, column=0, sticky="nsew")
        self.task_listbox.bind("<<ListboxSelect>>", lambda e: self._on_task_selected_tk())
        self.task_detail = scrolledtext.ScrolledText(task_frame, wrap="word", state="disabled",
                                                     height=8, font=("Consolas", 9))
        self.task_detail.grid(row=0, column=1, sticky="nsew", padx=(8, 0))

        sp_frame = tk.LabelFrame(tab, text="Scratchpad", padx=10, pady=10)
        sp_frame.grid(row=2, column=0, sticky="nsew")
        sp_frame.rowconfigure(0, weight=1)
        sp_frame.columnconfigure(0, weight=1)
        sp_frame.columnconfigure(1, weight=2)
        self.scratch_listbox = tk.Listbox(sp_frame, font=("Consolas", 9))
        self.scratch_listbox.grid(row=0, column=0, sticky="nsew")
        self.scratch_listbox.bind("<<ListboxSelect>>", lambda e: self._on_scratch_selected_tk())
        self.scratch_detail = scrolledtext.ScrolledText(sp_frame, wrap="word", state="disabled",
                                                        height=8, font=("Consolas", 9))
        self.scratch_detail.grid(row=0, column=1, sticky="nsew", padx=(8, 0))

        self._tasks_cache = []
        self._scratch_cache = []
        self.refresh_tasks_tab()

    def refresh_tasks_tab(self):
        """Reload TaskPad tasks + Scratchpad entries from disk."""
        try:
            self._tasks_cache = TaskPad().list_tasks()
        except Exception as exc:
            self._tasks_cache = []
            self._set_tk_detail(self.task_detail, f"Failed to load TaskPad: {exc}")
        self.task_listbox.delete(0, "end")
        for t in self._tasks_cache:
            status = t.status.value if hasattr(t.status, "value") else str(t.status)
            ttype = t.type.value if hasattr(t.type, "value") else str(t.type)
            prompt = (t.prompt or "")[:80].replace("\n", " ")
            self.task_listbox.insert("end", f"[{status}] {ttype} {t.id}: {prompt}")
        if self._tasks_cache:
            self.task_listbox.selection_set(0)
            self._on_task_selected_tk()
        else:
            self._set_tk_detail(self.task_detail, "No tasks.")

        try:
            self._scratch_cache = Scratchpad().get_recent(200)
        except Exception as exc:
            self._scratch_cache = []
            self._set_tk_detail(self.scratch_detail, f"Failed to load Scratchpad: {exc}")
        self.scratch_listbox.delete(0, "end")
        for e in self._scratch_cache:
            thought = (e.thought or "")[:80].replace("\n", " ")
            self.scratch_listbox.insert("end", f"[{e.model}] step {e.step}: {thought}")
        if self._scratch_cache:
            self.scratch_listbox.selection_set("end")
            self._on_scratch_selected_tk()
        else:
            self._set_tk_detail(self.scratch_detail, "No entries.")

    @staticmethod
    def _set_tk_detail(widget, text):
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.insert("1.0", text)
        widget.configure(state="disabled")

    def _on_task_selected_tk(self):
        sel = self.task_listbox.curselection()
        if not sel or sel[0] >= len(self._tasks_cache):
            return
        t = self._tasks_cache[sel[0]]
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
        self._set_tk_detail(self.task_detail, "\n".join(lines))

    def _on_scratch_selected_tk(self):
        sel = self.scratch_listbox.curselection()
        if not sel or sel[0] >= len(self._scratch_cache):
            return
        e = self._scratch_cache[sel[0]]
        self._set_tk_detail(
            self.scratch_detail,
            f"[{e.model}] step {e.step}\nthought: {e.thought}\nresult: {e.result or '—'}")
