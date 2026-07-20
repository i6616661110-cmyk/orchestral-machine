"""Orchestral Machine — Docker Sandbox.

Isolated Docker container for executing arbitrary commands.
One instance per task_id. Lifecycle is managed by execution_engine.py.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path

import docker
import docker.errors

logger = logging.getLogger(__name__)

SANDBOX_IMAGE = "orchestral-worker:latest"
SANDBOX_WORKSPACE_BASE = "./workspaces"

_active_sandboxes: dict[str, "DockerSandbox"] = {}
_registry_lock = threading.Lock()


def get_sandbox(task_id: str) -> "DockerSandbox | None":
    with _registry_lock:
        return _active_sandboxes.get(task_id)


def register_sandbox(task_id: str, sandbox: "DockerSandbox") -> None:
    with _registry_lock:
        _active_sandboxes[task_id] = sandbox


def unregister_sandbox(task_id: str) -> None:
    with _registry_lock:
        _active_sandboxes.pop(task_id, None)


class DockerSandbox:
    """Isolated Docker container for executing arbitrary commands.

    One instance per task_id. Lifecycle is managed by execution_engine.py.
    Retrieved by other modules via get_sandbox(task_id).
    """

    def __init__(
        self, task_id: str, workspace_base: str = SANDBOX_WORKSPACE_BASE
    ) -> None:
        self.task_id = task_id
        self.workspace_path = Path(workspace_base).resolve() / task_id
        self.workspace_path.mkdir(parents=True, exist_ok=True)
        self._client: docker.DockerClient | None = None
        self._container = None

    def start(self) -> None:
        self._client = docker.from_env()
        self._container = self._client.containers.run(
            SANDBOX_IMAGE,
            command="tail -f /dev/null",
            volumes={str(self.workspace_path): {"bind": "/workspace", "mode": "rw"}},
            working_dir="/workspace",
            detach=True,
            auto_remove=False,
            network_disabled=True,
        )
        logger.info(
            "Sandbox started: task=%s container=%s",
            self.task_id,
            self._container.short_id,
        )

    def stop(self) -> None:
        if self._container is not None:
            try:
                self._container.stop(timeout=5)
                self._container.remove(force=True)
            except Exception as exc:
                logger.warning(
                    "Error stopping sandbox for task %s: %s",
                    self.task_id,
                    exc,
                )
            self._container = None
        self._client = None

    def write_files(self, code: dict[str, str]) -> None:
        for filepath, content in code.items():
            target = self.workspace_path / filepath
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")

    def execute(self, cmd: str, timeout: int = 60) -> tuple[int, str, str]:
        if self._container is None:
            raise RuntimeError("Sandbox not started. Call start() first.")
        container = self._container

        result: list = [None]

        def _run() -> None:
            result[0] = container.exec_run(cmd, workdir="/workspace", demux=True)

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()
        thread.join(timeout=timeout)

        if thread.is_alive():
            logger.warning("Command timed out after %ds: %s", timeout, cmd)
            return (124, "", f"Execution timed out after {timeout}s")

        exit_code, output = result[0]
        stdout = (
            output[0].decode("utf-8", errors="replace") if output and output[0] else ""
        )
        stderr = (
            output[1].decode("utf-8", errors="replace") if output and output[1] else ""
        )
        return (exit_code or 0, stdout, stderr)

    def __enter__(self) -> "DockerSandbox":
        self.start()
        return self

    def __exit__(self, *args: object) -> None:
        self.stop()
