"""AI Scratchpad and Task Pad System.

Provides a shared scratchpad for models to think through problems step by step,
and a task pad for managing and executing tasks.
Stdlib only.
"""

import time
import json
import threading
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Callable
from pathlib import Path
from enum import Enum


SCRATCHPAD_PATH = "scratchpad.jsonl"
TASKPAD_PATH = "taskpad.jsonl"


class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskType(Enum):
    GENERATE = "generate"
    CODE = "code"
    ANALYZE = "analyze"
    REASON = "reason"
    TRAIN = "train"
    EMBED = "embed"
    SEARCH = "search"
    CUSTOM = "custom"


@dataclass
class ScratchpadEntry:
    """A single entry in the scratchpad."""
    step: int
    thought: str
    result: str
    model: str
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_jsonl(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_jsonl(cls, line: str) -> "ScratchpadEntry":
        data = json.loads(line)
        return cls(**data)


@dataclass
class Task:
    """A task to be executed."""
    id: str
    type: TaskType
    prompt: str
    model_name: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    status: TaskStatus = TaskStatus.PENDING
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    result: Optional[str] = None
    error: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)  # Task IDs this depends on
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_jsonl(self) -> str:
        data = asdict(self)
        data['type'] = self.type.value
        data['status'] = self.status.value
        return json.dumps(data, ensure_ascii=False)

    @classmethod
    def from_jsonl(cls, line: str) -> "Task":
        data = json.loads(line)
        data['type'] = TaskType(data['type'])
        data['status'] = TaskStatus(data['status'])
        return cls(**data)


class Scratchpad:
    """Shared scratchpad for collaborative thinking across models."""

    def __init__(self, path: str = SCRATCHPAD_PATH):
        self.path = Path(path)
        self._entries: List[ScratchpadEntry] = []
        self._lock = threading.Lock()
        self._callbacks: List[Callable[[ScratchpadEntry], None]] = []
        self._load()

    def _load(self) -> None:
        """Load scratchpad from file."""
        if self.path.exists():
            with open(self.path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = ScratchpadEntry.from_jsonl(line)
                        self._entries.append(entry)
                    except Exception:
                        continue

    def _save(self) -> None:
        """Save scratchpad to file."""
        with open(self.path, "w", encoding="utf-8") as f:
            for entry in self._entries:
                f.write(entry.to_jsonl() + "\n")

    def add(self, thought: str, result: str, model: str, 
            step: Optional[int] = None, metadata: Optional[Dict] = None) -> ScratchpadEntry:
        """Add a new entry to the scratchpad."""
        with self._lock:
            if step is None:
                step = len(self._entries)
            entry = ScratchpadEntry(
                step=step,
                thought=thought,
                result=result,
                model=model,
                metadata=metadata or {}
            )
            self._entries.append(entry)
            self._save()

        # Trigger callbacks
        for cb in self._callbacks:
            try:
                cb(entry)
            except Exception:
                pass

        return entry

    def get_recent(self, limit: int = 50) -> List[ScratchpadEntry]:
        """Get recent entries."""
        with self._lock:
            return self._entries[-limit:]

    def get_by_model(self, model: str, limit: int = 50) -> List[ScratchpadEntry]:
        """Get entries by model."""
        with self._lock:
            return [e for e in reversed(self._entries) if e.model == model][:limit]

    def get_context(self, max_entries: int = 10) -> str:
        """Get formatted context for model input."""
        with self._lock:
            recent = self._entries[-max_entries:]
            lines = []
            for e in recent:
                lines.append(f"[{e.model}] Step {e.step}: {e.thought}")
                if e.result:
                    lines.append(f"  -> {e.result}")
            return "\n".join(lines)

    def clear(self) -> None:
        """Clear the scratchpad."""
        with self._lock:
            self._entries.clear()
            self._save()

    def register_callback(self, callback: Callable[[ScratchpadEntry], None]) -> None:
        """Register callback for new entries."""
        self._callbacks.append(callback)


class TaskPad:
    """Task management and execution system."""

    def __init__(self, path: str = TASKPAD_PATH):
        self.path = Path(path)
        self._tasks: Dict[str, Task] = {}
        self._lock = threading.Lock()
        self._executors: Dict[TaskType, Callable] = {}
        self._load()

    def register_executor(self, task_type: TaskType, executor: Callable[[Task], str]) -> None:
        """Register an executor function for a task type."""
        self._executors[task_type] = executor

    def _load(self) -> None:
        """Load tasks from file."""
        if self.path.exists():
            with open(self.path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        task = Task.from_jsonl(line)
                        self._tasks[task.id] = task
                    except Exception:
                        continue

    def _save(self) -> None:
        """Save tasks to file."""
        with open(self.path, "w", encoding="utf-8") as f:
            for task in self._tasks.values():
                f.write(task.to_jsonl() + "\n")

    def create_task(self, type: TaskType, prompt: str, 
                    model_name: Optional[str] = None,
                    parameters: Optional[Dict] = None,
                    dependencies: Optional[List[str]] = None) -> Task:
        """Create a new task."""
        task_id = f"task_{int(time.time() * 1000000)}"
        task = Task(
            id=task_id,
            type=type,
            prompt=prompt,
            model_name=model_name,
            parameters=parameters or {},
            dependencies=dependencies or []
        )
        with self._lock:
            self._tasks[task_id] = task
            self._save()
        return task

    def get_task(self, task_id: str) -> Optional[Task]:
        """Get task by ID."""
        with self._lock:
            return self._tasks.get(task_id)

    def list_tasks(self, status: Optional[TaskStatus] = None) -> List[Task]:
        """List all tasks, optionally filtered by status."""
        with self._lock:
            tasks = list(self._tasks.values())
            if status:
                tasks = [t for t in tasks if t.status == status]
            tasks.sort(key=lambda t: t.created_at, reverse=True)
            return tasks

    def execute_task(self, task_id: str) -> Task:
        """Execute a single task."""
        with self._lock:
            task = self._tasks.get(task_id)
            if not task:
                raise KeyError(f"Task {task_id} not found")
            
            if task.status != TaskStatus.PENDING:
                raise ValueError(f"Task {task_id} is not pending (status: {task.status.value})")

            # Check dependencies
            for dep_id in task.dependencies:
                dep = self._tasks.get(dep_id)
                if not dep or dep.status != TaskStatus.COMPLETED:
                    raise ValueError(f"Dependency {dep_id} not completed")

            task.status = TaskStatus.RUNNING
            task.started_at = time.time()
            self._save()

        try:
            # Get executor
            executor = self._executors.get(task.type)
            if not executor:
                raise ValueError(f"No executor registered for task type {task.type.value}")

            # Execute
            result = executor(task)
            task.result = result
            task.status = TaskStatus.COMPLETED
        except Exception as e:
            task.error = str(e)
            task.status = TaskStatus.FAILED
        finally:
            task.completed_at = time.time()
            with self._lock:
                self._save()

        return task

    def execute_all_pending(self) -> List[Task]:
        """Execute all pending tasks in dependency order."""
        # Simple topological sort based on dependencies
        with self._lock:
            pending = [t for t in self._tasks.values() if t.status == TaskStatus.PENDING]
        
        executed = []
        while pending:
            progress = False
            for task in pending[:]:
                deps_met = all(
                    self._tasks.get(d, Task("", TaskType.CUSTOM, "")).status == TaskStatus.COMPLETED
                    for d in task.dependencies
                )
                if deps_met:
                    try:
                        self.execute_task(task.id)
                        executed.append(task)
                        progress = True
                    except Exception:
                        pass
            if not progress:
                # Circular dependency or missing executor
                break
        return executed


class ThinkEngine:
    """High-level thinking engine that uses scratchpad for step-by-step reasoning."""

    def __init__(self, scratchpad: Scratchpad, model_loader: Callable[[str], Any],
                 default_model: str = "tiny-prose"):
        self.scratchpad = scratchpad
        self.model_loader = model_loader
        self.default_model = default_model

    def think(self, prompt: str, steps: int = 3, model_name: Optional[str] = None) -> Dict[str, Any]:
        """Think through a problem step by step using the scratchpad.

        Args:
            prompt: The problem to think about
            steps: Number of thinking steps
            model_name: Model to use (default: self.default_model)

        Returns:
            Dict with 'answer', 'trace', 'scratchpad_entries'
        """
        model_name = model_name or self.default_model
        model = self.model_loader(model_name)
        
        # Initial context
        context = f"Problem: {prompt}\n\nLet me think through this step by step:\n"
        
        trace = []
        
        for i in range(steps):
            step_num = i + 1
            
            # Generate thought
            thought_prompt = f"{context}Step {step_num}: "
            # Use model to generate thought (simplified - in practice would use generate())
            # For now, we'll use a template-based approach
            thought = self._generate_thought(model, thought_prompt, i, steps)
            
            # Get result from model
            result_prompt = f"{context}{thought}\nResult: "
            result = self._generate_result(model, result_prompt)
            
            # Add to scratchpad
            entry = self.scratchpad.add(
                thought=thought,
                result=result,
                model=model_name,
                step=i,
                metadata={"step_type": "thinking"}
            )
            
            trace.append({
                "step": step_num,
                "thought": thought,
                "result": result
            })
            
            # Update context
            context += f"Step {step_num}: {thought}\nResult: {result}\n\n"
        
        # Final answer
        final_prompt = f"{context}Final Answer: "
        final_answer = self._generate_result(model, final_prompt)
        
        # Add final answer to scratchpad
        self.scratchpad.add(
            thought="Synthesizing final answer",
            result=final_answer,
            model=model_name,
            step=steps,
            metadata={"step_type": "final_answer"}
        )
        
        return {
            "answer": final_answer,
            "trace": trace,
            "scratchpad_entries": len(self.scratchpad.get_recent())
        }

    def _generate_thought(self, model, prompt: str, step: int, total: int) -> str:
        """Generate a thinking step. Override with model-specific logic."""
        # Template-based thinking for different domains
        templates = {
            0: "First, I need to understand what is being asked. The key elements are:",
            1: "Now I'll break this down into smaller parts:",
            2: "Let me consider the implications and connections:",
        }
        template = templates.get(step, "Continuing the analysis:")
        return f"{template} {prompt[:100]}"

    def _generate_result(self, model, prompt: str) -> str:
        """Generate result from model. Override with actual model generation."""
        # In practice, this would call model.generate()
        return f"[Generated by {type(model).__name__}]"


# Global instances
SCRATCHPAD = Scratchpad()
TASKPAD = TaskPad()


def create_scratchpad_system() -> tuple:
    """Create a complete scratchpad system."""
    return SCRATCHPAD, TaskPad()


if __name__ == "__main__":
    # Demo
    scratchpad = Scratchpad()
    taskpad = TaskPad()

    # Add some scratchpad entries
    scratchpad.add("Understanding the problem", "Need to write a Python function", "tiny-code")
    scratchpad.add("Breaking down", "Function takes list, returns sum", "tiny-code")
    scratchpad.add("Final answer", "def sum_list(lst): return sum(lst)", "tiny-code")

    print("Scratchpad context:")
    print(scratchpad.get_context())

    # Create tasks
    task1 = taskpad.create_task(TaskType.CODE, "Write a hello world function", "tiny-code")
    task2 = taskpad.create_task(TaskType.ANALYZE, "Analyze code complexity", "tiny-reason", dependencies=[task1.id])

    print(f"\nCreated tasks: {task1.id}, {task2.id}")
    print("All tasks:", [t.id for t in taskpad.list_tasks()])