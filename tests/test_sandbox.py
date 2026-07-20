from unittest.mock import MagicMock, patch

import pytest

from src.sandbox import (
    SANDBOX_IMAGE,
    DockerSandbox,
    get_sandbox,
    register_sandbox,
    unregister_sandbox,
)


class TestRegistry:
    def test_register_and_get(self):
        tid = "task-reg-1"
        sb = MagicMock(spec=DockerSandbox)
        try:
            register_sandbox(tid, sb)
            assert get_sandbox(tid) is sb
        finally:
            unregister_sandbox(tid)

    def test_get_unknown_returns_none(self):
        assert get_sandbox("nonexistent") is None

    def test_unregister_removes_entry(self):
        tid = "task-reg-2"
        sb = MagicMock(spec=DockerSandbox)
        register_sandbox(tid, sb)
        unregister_sandbox(tid)
        assert get_sandbox(tid) is None

    def test_unregister_unknown_is_safe(self):
        unregister_sandbox("nonexistent")

    def test_register_overwrites(self):
        tid = "task-reg-3"
        first = MagicMock(spec=DockerSandbox)
        second = MagicMock(spec=DockerSandbox)
        try:
            register_sandbox(tid, first)
            register_sandbox(tid, second)
            assert get_sandbox(tid) is second
        finally:
            unregister_sandbox(tid)


class TestDockerSandboxWriteFiles:
    def test_single_file(self, tmp_path):
        sandbox = DockerSandbox(task_id="test", workspace_base=str(tmp_path))
        sandbox.write_files({"main.py": "print(1)"})

        target = sandbox.workspace_path / "main.py"
        assert target.exists()
        assert target.read_text(encoding="utf-8") == "print(1)"

    def test_nested_path(self, tmp_path):
        sandbox = DockerSandbox(task_id="test", workspace_base=str(tmp_path))
        sandbox.write_files({"src/utils.py": "x=1"})

        target = sandbox.workspace_path / "src" / "utils.py"
        assert target.parent.exists()
        assert target.read_text(encoding="utf-8") == "x=1"

    def test_multiple_files(self, tmp_path):
        sandbox = DockerSandbox(task_id="test", workspace_base=str(tmp_path))
        code = {
            "main.py": "print('ok')",
            "src/a.py": "a=1",
            "tests/test_a.py": "def test_a():\n    assert True",
        }
        sandbox.write_files(code)

        assert (sandbox.workspace_path / "main.py").exists()
        assert (sandbox.workspace_path / "src" / "a.py").exists()
        assert (sandbox.workspace_path / "tests" / "test_a.py").exists()

    def test_overwrites_existing(self, tmp_path):
        sandbox = DockerSandbox(task_id="test", workspace_base=str(tmp_path))
        sandbox.write_files({"main.py": "first"})
        sandbox.write_files({"main.py": "second"})

        target = sandbox.workspace_path / "main.py"
        assert target.read_text(encoding="utf-8") == "second"


class TestDockerSandboxLifecycle:
    @patch("src.sandbox.docker.from_env")
    def test_start_calls_containers_run(self, mock_from_env, tmp_path):
        mock_client = MagicMock()
        mock_container = MagicMock()
        mock_container.short_id = "abc123"
        mock_client.containers.run.return_value = mock_container
        mock_from_env.return_value = mock_client

        sandbox = DockerSandbox(task_id="test", workspace_base=str(tmp_path))
        sandbox.start()

        mock_client.containers.run.assert_called_once()
        args, kwargs = mock_client.containers.run.call_args
        assert args[0] == SANDBOX_IMAGE
        assert kwargs["detach"] is True
        assert kwargs["auto_remove"] is False
        assert kwargs["network_disabled"] is True

    @patch("src.sandbox.docker.from_env")
    def test_start_sets_container(self, mock_from_env, tmp_path):
        mock_client = MagicMock()
        mock_container = MagicMock()
        mock_container.short_id = "abc123"
        mock_client.containers.run.return_value = mock_container
        mock_from_env.return_value = mock_client

        sandbox = DockerSandbox(task_id="test", workspace_base=str(tmp_path))
        sandbox.start()

        assert sandbox._container is mock_container

    @patch("src.sandbox.docker.from_env")
    def test_stop_calls_stop_and_remove(self, mock_from_env, tmp_path):
        mock_client = MagicMock()
        mock_container = MagicMock()
        mock_container.short_id = "abc123"
        mock_client.containers.run.return_value = mock_container
        mock_from_env.return_value = mock_client

        sandbox = DockerSandbox(task_id="test", workspace_base=str(tmp_path))
        sandbox.start()
        sandbox.stop()

        mock_container.stop.assert_called_once_with(timeout=5)
        mock_container.remove.assert_called_once_with(force=True)

    @patch("src.sandbox.docker.from_env")
    def test_stop_clears_container(self, mock_from_env, tmp_path):
        mock_client = MagicMock()
        mock_container = MagicMock()
        mock_container.short_id = "abc123"
        mock_client.containers.run.return_value = mock_container
        mock_from_env.return_value = mock_client

        sandbox = DockerSandbox(task_id="test", workspace_base=str(tmp_path))
        sandbox.start()
        sandbox.stop()

        assert sandbox._container is None
        assert sandbox._client is None

    def test_stop_before_start_is_safe(self, tmp_path):
        sandbox = DockerSandbox(task_id="test", workspace_base=str(tmp_path))
        sandbox.stop()

    @patch("src.sandbox.docker.from_env")
    def test_stop_handles_container_error(self, mock_from_env, tmp_path):
        mock_client = MagicMock()
        mock_container = MagicMock()
        mock_container.short_id = "abc123"
        mock_container.stop.side_effect = Exception("boom")
        mock_client.containers.run.return_value = mock_container
        mock_from_env.return_value = mock_client

        sandbox = DockerSandbox(task_id="test", workspace_base=str(tmp_path))
        sandbox.start()
        sandbox.stop()

        assert sandbox._container is None
        assert sandbox._client is None

    @patch("src.sandbox.docker.from_env")
    def test_context_manager(self, mock_from_env, tmp_path):
        mock_client = MagicMock()
        mock_container = MagicMock()
        mock_container.short_id = "abc123"
        mock_client.containers.run.return_value = mock_container
        mock_from_env.return_value = mock_client

        sandbox = DockerSandbox(task_id="test", workspace_base=str(tmp_path))
        with sandbox as active:
            assert active is sandbox
            assert sandbox._container is mock_container

        mock_client.containers.run.assert_called_once()
        mock_container.stop.assert_called_once_with(timeout=5)
        mock_container.remove.assert_called_once_with(force=True)


class TestDockerSandboxExecute:
    @patch("src.sandbox.docker.from_env")
    def test_execute_returns_stdout_stderr(self, mock_from_env, tmp_path):
        mock_client = MagicMock()
        mock_container = MagicMock()
        mock_container.short_id = "abc123"
        mock_container.exec_run.return_value = (0, (b"hello", b"warn"))
        mock_client.containers.run.return_value = mock_container
        mock_from_env.return_value = mock_client

        sandbox = DockerSandbox(task_id="test", workspace_base=str(tmp_path))
        sandbox.start()
        code, stdout, stderr = sandbox.execute("echo hello")

        assert code == 0
        assert stdout == "hello"
        assert stderr == "warn"

    @patch("src.sandbox.docker.from_env")
    def test_execute_exit_code_zero(self, mock_from_env, tmp_path):
        mock_client = MagicMock()
        mock_container = MagicMock()
        mock_container.short_id = "abc123"
        mock_container.exec_run.return_value = (0, (b"ok", b""))
        mock_client.containers.run.return_value = mock_container
        mock_from_env.return_value = mock_client

        sandbox = DockerSandbox(task_id="test", workspace_base=str(tmp_path))
        sandbox.start()
        code, _, _ = sandbox.execute("true")

        assert code == 0

    @patch("src.sandbox.docker.from_env")
    def test_execute_exit_code_nonzero(self, mock_from_env, tmp_path):
        mock_client = MagicMock()
        mock_container = MagicMock()
        mock_container.short_id = "abc123"
        mock_container.exec_run.return_value = (1, (b"", b"err"))
        mock_client.containers.run.return_value = mock_container
        mock_from_env.return_value = mock_client

        sandbox = DockerSandbox(task_id="test", workspace_base=str(tmp_path))
        sandbox.start()
        code, _, _ = sandbox.execute("false")

        assert code == 1

    @patch("src.sandbox.docker.from_env")
    def test_execute_none_output_parts(self, mock_from_env, tmp_path):
        mock_client = MagicMock()
        mock_container = MagicMock()
        mock_container.short_id = "abc123"
        mock_container.exec_run.return_value = (0, (None, None))
        mock_client.containers.run.return_value = mock_container
        mock_from_env.return_value = mock_client

        sandbox = DockerSandbox(task_id="test", workspace_base=str(tmp_path))
        sandbox.start()
        code, stdout, stderr = sandbox.execute("echo")

        assert code == 0
        assert stdout == ""
        assert stderr == ""

    @patch("src.sandbox.docker.from_env")
    def test_execute_none_output_tuple(self, mock_from_env, tmp_path):
        mock_client = MagicMock()
        mock_container = MagicMock()
        mock_container.short_id = "abc123"
        mock_container.exec_run.return_value = (0, None)
        mock_client.containers.run.return_value = mock_container
        mock_from_env.return_value = mock_client

        sandbox = DockerSandbox(task_id="test", workspace_base=str(tmp_path))
        sandbox.start()
        code, stdout, stderr = sandbox.execute("echo")

        assert code == 0
        assert stdout == ""
        assert stderr == ""

    def test_execute_before_start_raises(self, tmp_path):
        sandbox = DockerSandbox(task_id="test", workspace_base=str(tmp_path))
        with pytest.raises(RuntimeError):
            sandbox.execute("echo hello")

    @patch("src.sandbox.docker.from_env")
    def test_execute_passes_cmd_and_workdir(self, mock_from_env, tmp_path):
        mock_client = MagicMock()
        mock_container = MagicMock()
        mock_container.short_id = "abc123"
        mock_container.exec_run.return_value = (0, (b"ok", b""))
        mock_client.containers.run.return_value = mock_container
        mock_from_env.return_value = mock_client

        sandbox = DockerSandbox(task_id="test", workspace_base=str(tmp_path))
        sandbox.start()
        sandbox.execute("python main.py")

        mock_container.exec_run.assert_called_once_with(
            "python main.py",
            workdir="/workspace",
            demux=True,
        )


class TestDockerSandboxTimeout:
    @patch("src.sandbox.docker.from_env")
    def test_timeout_returns_124(self, mock_from_env, tmp_path):
        """Command that exceeds timeout returns exit code 124."""
        import time

        mock_client = MagicMock()
        mock_container = MagicMock()
        mock_container.short_id = "abc123"

        def slow_exec(*args, **kwargs):
            time.sleep(5)
            return (0, (b"ok", b""))

        mock_container.exec_run.side_effect = slow_exec
        mock_client.containers.run.return_value = mock_container
        mock_from_env.return_value = mock_client

        sandbox = DockerSandbox(task_id="test", workspace_base=str(tmp_path))
        sandbox.start()
        code, stdout, stderr = sandbox.execute("sleep 100", timeout=1)

        assert code == 124
        assert stdout == ""
        assert "timed out" in stderr

    @patch("src.sandbox.docker.from_env")
    def test_fast_command_unaffected_by_timeout(self, mock_from_env, tmp_path):
        """Commands finishing before timeout work normally."""
        mock_client = MagicMock()
        mock_container = MagicMock()
        mock_container.short_id = "abc123"
        mock_container.exec_run.return_value = (0, (b"hello", b""))
        mock_client.containers.run.return_value = mock_container
        mock_from_env.return_value = mock_client

        sandbox = DockerSandbox(task_id="test", workspace_base=str(tmp_path))
        sandbox.start()
        code, stdout, stderr = sandbox.execute("echo hello", timeout=30)

        assert code == 0
        assert stdout == "hello"
        assert stderr == ""

    @patch("src.sandbox.docker.from_env")
    def test_default_timeout_is_60(self, mock_from_env, tmp_path):
        """Default timeout parameter is 60 seconds."""
        import inspect

        sig = inspect.signature(DockerSandbox.execute)
        assert sig.parameters["timeout"].default == 60
