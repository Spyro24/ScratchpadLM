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


# =============================================================================
# Harness: Automation Engine for System Interaction
# =============================================================================

class HarnessPrivilege(Enum):
    """Privilege levels for harness commands."""
    READ_ONLY = "read_only"      # Can only read files, no execution
    READ_WRITE = "read_write"    # Can read/write files, no shell/code
    FULL_ACCESS = "full_access"  # Can execute code, shell commands
    ADMIN = "admin"              # Full access + model management


@dataclass
class HarnessConfig:
    """Configuration for harness execution."""
    privilege: HarnessPrivilege = HarnessPrivilege.READ_ONLY
    root_path: str = "."         # Working directory root (sandboxed)
    timeout: float = 30.0        # Default command timeout (seconds)
    max_output_size: int = 10000 # Max output characters per command


class HarnessCommand:
    """Represents a single harness command."""
    
    def __init__(self, name: str, args: List[str], kwargs: Dict[str, Any], line_num: int = 0):
        self.name = name
        self.args = args
        self.kwargs = kwargs
        self.line_num = line_num
    
    def __repr__(self):
        return f"HarnessCommand({self.name}, args={self.args}, kwargs={self.kwargs})"


class HarnessRuntime:
    """Harness runtime for executing .tlm scripts."""
    
    def __init__(self, config: HarnessConfig = None):
        self.config = config or HarnessConfig()
        self.variables: Dict[str, Any] = {}
        self._command_handlers: Dict[str, Callable] = {}
        self._register_builtin_commands()
    
    def _register_builtin_commands(self):
        """Register built-in command handlers."""
        self._command_handlers = {
            "prompt": self._cmd_prompt,
            "task": self._cmd_task,
            "assert": self._cmd_assert,
            "loop": self._cmd_loop,
            "branch": self._cmd_branch,
            "parallel": self._cmd_parallel,
            "retry": self._cmd_retry,
            "save": self._cmd_save,
            "load": self._cmd_load,
            "code_run": self._cmd_code_run,
            "shell": self._cmd_shell,
            "file_read": self._cmd_file_read,
            "file_write": self._cmd_file_write,
            "http_request": self._cmd_http_request,
            "git": self._cmd_git,
            "search": self._cmd_search,
            "model_load": self._cmd_model_load,
            "model_create": self._cmd_model_create,
        }
    
    def _check_privilege(self, required: HarnessPrivilege) -> bool:
        """Check if current privilege level allows the operation."""
        levels = {
            HarnessPrivilege.READ_ONLY: 0,
            HarnessPrivilege.READ_WRITE: 1,
            HarnessPrivilege.FULL_ACCESS: 2,
            HarnessPrivilege.ADMIN: 3,
        }
        return levels[self.config.privilege] >= levels[required]
    
    def _resolve_path(self, path: str) -> Path:
        """Resolve path within sandbox root."""
        root = Path(self.config.root_path).resolve()
        target = (root / path).resolve()
        if not str(target).startswith(str(root)):
            raise PermissionError(f"Path '{path}' escapes sandbox root")
        return target
    
    def _interpolate(self, text: str) -> str:
        """Interpolate variables in text: ${var} or $var."""
        import re
        def replace_var(match):
            var_name = match.group(1)
            return str(self.variables.get(var_name, f"${{{var_name}}}"))
        return re.sub(r'\$\{([^}]+)\}', replace_var, text)
    
    # ---- Built-in command handlers ----
    
    def _cmd_prompt(self, args: List[str], kwargs: Dict) -> Any:
        """prompt "text" - Send prompt to model and capture response."""
        if not args:
            raise ValueError("prompt requires a text argument")
        prompt = self._interpolate(args[0])
        model_name = kwargs.get("model", self.variables.get("_model", "tiny-prose"))
        # In real implementation, would call model.generate()
        result = f"[Response from {model_name} to: {prompt[:50]}...]"
        self.variables["_last_response"] = result
        return result
    
    def _cmd_task(self, args: List[str], kwargs: Dict) -> Any:
        """task "type" "prompt" - Create and execute a task."""
        if len(args) < 2:
            raise ValueError("task requires type and prompt arguments")
        task_type = TaskType(args[0])
        prompt = self._interpolate(args[1])
        model = kwargs.get("model", self.variables.get("_model"))
        task = TASKPAD.create_task(task_type, prompt, model)
        return TASKPAD.execute_task(task.id)
    
    def _cmd_assert(self, args: List[str], kwargs: Dict) -> Any:
        """assert condition - Assert a condition, fail if false."""
        if not args:
            raise ValueError("assert requires a condition")
        condition = self._interpolate(args[0])
        # Simple evaluation - in practice would be more sophisticated
        result = eval(condition, {"__builtins__": {}}, self.variables)
        if not result:
            raise AssertionError(f"Assertion failed: {condition}")
        return True
    
    def _cmd_loop(self, args: List[str], kwargs: Dict) -> Any:
        """loop N { commands } - Execute commands N times."""
        if len(args) < 2:
            raise ValueError("loop requires count and body")
        count = int(self._interpolate(args[0]))
        body = args[1]
        results = []
        for i in range(count):
            self.variables["_loop_index"] = i
            results.append(self.execute_script(body))
        return results
    
    def _cmd_branch(self, args: List[str], kwargs: Dict) -> Any:
        """branch condition { if_body } { else_body } - Conditional execution."""
        if len(args) < 2:
            raise ValueError("branch requires condition and bodies")
        condition = self._interpolate(args[0])
        if_body = args[1]
        else_body = args[2] if len(args) > 2 else ""
        result = eval(condition, {"__builtins__": {}}, self.variables)
        if result:
            return self.execute_script(if_body)
        elif else_body:
            return self.execute_script(else_body)
        return None
    
    def _cmd_parallel(self, args: List[str], kwargs: Dict) -> Any:
        """parallel [cmd1, cmd2, ...] - Execute commands in parallel."""
        import concurrent.futures
        results = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(self.execute_script, cmd) for cmd in args]
            for f in concurrent.futures.as_completed(futures):
                results.append(f.result())
        return results
    
    def _cmd_retry(self, args: List[str], kwargs: Dict) -> Any:
        """retry N { command } - Retry command up to N times with backoff."""
        if len(args) < 2:
            raise ValueError("retry requires count and command")
        max_attempts = int(self._interpolate(args[0]))
        cmd = args[1]
        for attempt in range(max_attempts):
            try:
                return self.execute_script(cmd)
            except Exception as e:
                if attempt == max_attempts - 1:
                    raise
                time.sleep(2 ** attempt)  # Exponential backoff
        return None
    
    def _cmd_save(self, args: List[str], kwargs: Dict) -> Any:
        """save var_name - Save variable to file."""
        if not args:
            raise ValueError("save requires variable name")
        var_name = args[0]
        path = kwargs.get("path", f"{var_name}.json")
        target = self._resolve_path(path)
        import json
        target.write_text(json.dumps(self.variables.get(var_name)), encoding="utf-8")
        return True
    
    def _cmd_load(self, args: List[str], kwargs: Dict) -> Any:
        """load var_name - Load variable from file."""
        if not args:
            raise ValueError("load requires variable name")
        var_name = args[0]
        path = kwargs.get("path", f"{var_name}.json")
        target = self._resolve_path(path)
        import json
        self.variables[var_name] = json.loads(target.read_text(encoding="utf-8"))
        return self.variables[var_name]
    
    def _cmd_code_run(self, args: List[str], kwargs: Dict) -> Any:
        """code_run "code" - Execute Python code in sandbox."""
        if not self._check_privilege(HarnessPrivilege.FULL_ACCESS):
            raise PermissionError("code_run requires FULL_ACCESS privilege")
        if not args:
            raise ValueError("code_run requires code argument")
        code = self._interpolate(args[0])
        # WARNING: This executes arbitrary Python code
        # In production, use a proper sandbox (subprocess, docker, etc.)
        local_vars = {**self.variables}
        exec(code, {"__builtins__": __builtins__}, local_vars)
        self.variables.update(local_vars)
        return local_vars.get("_result")
    
    def _cmd_shell(self, args: List[str], kwargs: Dict) -> Any:
        """shell "command" - Execute shell command."""
        if not self._check_privilege(HarnessPrivilege.FULL_ACCESS):
            raise PermissionError("shell requires FULL_ACCESS privilege")
        if not args:
            raise ValueError("shell requires command argument")
        cmd = self._interpolate(args[0])
        import subprocess
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True,
            timeout=self.config.timeout, cwd=self.config.root_path
        )
        output = result.stdout[:self.config.max_output_size]
        if result.stderr:
            output += "\nSTDERR:\n" + result.stderr[:self.config.max_output_size]
        self.variables["_last_shell_result"] = output
        return output
    
    def _cmd_file_read(self, args: List[str], kwargs: Dict) -> Any:
        """file_read "path" - Read file contents."""
        if not self._check_privilege(HarnessPrivilege.READ_ONLY):
            raise PermissionError("file_read requires READ_ONLY privilege")
        if not args:
            raise ValueError("file_read requires path argument")
        path = self._resolve_path(self._interpolate(args[0]))
        return path.read_text(encoding="utf-8")
    
    def _cmd_file_write(self, args: List[str], kwargs: Dict) -> Any:
        """file_write "path" "content" - Write file."""
        if not self._check_privilege(HarnessPrivilege.READ_WRITE):
            raise PermissionError("file_write requires READ_WRITE privilege")
        if len(args) < 2:
            raise ValueError("file_write requires path and content arguments")
        path = self._resolve_path(self._interpolate(args[0]))
        content = self._interpolate(args[1])
        path.write_text(content, encoding="utf-8")
        return True
    
    def _cmd_http_request(self, args: List[str], kwargs: Dict) -> Any:
        """http_request "method" "url" - Make HTTP request."""
        if not self._check_privilege(HarnessPrivilege.FULL_ACCESS):
            raise PermissionError("http_request requires FULL_ACCESS privilege")
        if len(args) < 2:
            raise ValueError("http_request requires method and url")
        method = args[0].upper()
        url = self._interpolate(args[1])
        import urllib.request, urllib.parse
        data = kwargs.get("data")
        headers = kwargs.get("headers", {})
        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        with urllib.request.urlopen(req, timeout=self.config.timeout) as resp:
            return resp.read().decode("utf-8")[:self.config.max_output_size]
    
    def _cmd_git(self, args: List[str], kwargs: Dict) -> Any:
        """git "subcommand" [args...] - Run git command."""
        if not self._check_privilege(HarnessPrivilege.FULL_ACCESS):
            raise PermissionError("git requires FULL_ACCESS privilege")
        if not args:
            raise ValueError("git requires subcommand")
        cmd = ["git"] + [self._interpolate(a) for a in args]
        import subprocess
        result = subprocess.run(
            cmd, capture_output=True, text=True,
            timeout=self.config.timeout, cwd=self.config.root_path
        )
        return result.stdout[:self.config.max_output_size]
    
    def _cmd_search(self, args: List[str], kwargs: Dict) -> Any:
        """search "query" - Search web (placeholder)."""
        if not self._check_privilege(HarnessPrivilege.FULL_ACCESS):
            raise PermissionError("search requires FULL_ACCESS privilege")
        query = self._interpolate(args[0]) if args else ""
        return f"[Search results for: {query}]"
    
    def _cmd_model_load(self, args: List[str], kwargs: Dict) -> Any:
        """model_load "name" - Load model into variable."""
        if not self._check_privilege(HarnessPrivilege.ADMIN):
            raise PermissionError("model_load requires ADMIN privilege")
        if not args:
            raise ValueError("model_load requires model name")
        name = args[0]
        var_name = kwargs.get("as", "_model")
        # In real implementation, would load from catalog
        self.variables[var_name] = name
        return name
    
    def _cmd_model_create(self, args: List[str], kwargs: Dict) -> Any:
        """model_create "name" --domains "domain1,domain2" - Create new model."""
        if not self._check_privilege(HarnessPrivilege.ADMIN):
            raise PermissionError("model_create requires ADMIN privilege")
        if not args:
            raise ValueError("model_create requires model name")
        name = args[0]
        domains = kwargs.get("domains", "").split(",")
        domains = [d.strip() for d in domains if d.strip()]
        # In real implementation, would use ModelCreator
        self.variables["_last_created_model"] = name
        return f"Model {name} created with domains: {domains}"
    
    # ---- Script execution ----
    
    def execute_script(self, script: str) -> Any:
        """Execute a harness script (string)."""
        lines = script.strip().split('\n')
        results = []
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            i += 1
            if not line or line.startswith('#'):
                continue
            # Handle multi-line commands (simple brace counting)
            if '{' in line and '}' not in line:
                # Find matching closing brace
                brace_count = line.count('{') - line.count('}')
                body_lines = [line]
                while i < len(lines) and brace_count > 0:
                    next_line = lines[i]
                    brace_count += next_line.count('{') - next_line.count('}')
                    body_lines.append(next_line)
                    i += 1
                line = '\n'.join(body_lines)
            
            # Parse command
            cmd = self._parse_command(line, i)
            if cmd:
                try:
                    handler = self._command_handlers.get(cmd.name)
                    if not handler:
                        raise ValueError(f"Unknown command: {cmd.name}")
                    result = handler(cmd.args, cmd.kwargs)
                    results.append(result)
                except Exception as e:
                    raise RuntimeError(f"Line {cmd.line_num}: {e}")
        return results[-1] if results else None
    
    def _parse_command(self, line: str, line_num: int) -> HarnessCommand:
        """Parse a command line into HarnessCommand."""
        import shlex
        tokens = shlex.split(line)
        if not tokens:
            return None
        name = tokens[0]
        args = []
        kwargs = {}
        for token in tokens[1:]:
            if '=' in token and not token.startswith('"') and not token.startswith("'"):
                key, value = token.split('=', 1)
                kwargs[key] = value
            else:
                args.append(token)
        return HarnessCommand(name, args, kwargs, line_num)
    
    def run_file(self, path: str) -> Any:
        """Run a harness script from file."""
        target = self._resolve_path(path)
        script = target.read_text(encoding="utf-8")
        return self.execute_script(script)


def create_harness(config: HarnessConfig = None) -> HarnessRuntime:
    """Create a harness runtime instance."""
    return HarnessRuntime(config)


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