"""Behavioral tests for the run_ddp launcher script."""

import os
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest


class TestRunDdpScript:
    """Integration-style tests that exercise the Bash launcher in realistic scenarios."""

    @staticmethod
    def _install_torchrun_stub(
        tmp_path: Path,
    ) -> Path:
        """Create a shim torchrun binary that execs `python -m <module>`."""
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir(exist_ok=True)
        torchrun_stub_path = bin_dir / "torchrun"
        torchrun_stub_path.write_text(
            textwrap.dedent(
                """\
                #!/usr/bin/env bash
                set -euo pipefail
                module=""
                while [[ $# -gt 0 ]]; do
                  case "$1" in
                    --nproc_per_node|--nnodes|--node_rank|--master_port|--master_addr)
                      shift 2
                      ;;
                    --standalone|--rdzv_backend|--rdzv_endpoint|--rdzv_id)
                      shift
                      ;;
                    -m)
                      module="$2"
                      shift 2
                      break
                      ;;
                    *)
                      shift
                      ;;
                  esac
                done
                exec "${PYTHON_BIN:-python}" -m "${module}" "$@"
                """
            )
        )
        torchrun_stub_path.chmod(0o755)
        return bin_dir

    @staticmethod
    def _write_module(
        module_dir: Path,
        module_name: str,
        body: str,
    ) -> None:
        """Write a Python module to `module_dir/module_name.py`."""
        module_dir.mkdir(exist_ok=True)
        (module_dir / f"{module_name}.py").write_text(textwrap.dedent(body))

    def test_sigint_reaches_torchrun_worker(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Ensure that Ctrl+C propagation works when tee logging is enabled."""
        repo_root = Path(__file__).resolve().parents[2]
        launcher_path = repo_root / "scripts" / "run_ddp.sh"
        assert launcher_path.exists(), "launcher script should be present in the repo"

        bin_dir = self._install_torchrun_stub(tmp_path)
        module_dir = tmp_path / "fake_app"
        signal_record_path = tmp_path / "torchrun_signal.txt"
        ready_marker_path = tmp_path / "torchrun_ready.txt"

        self._write_module(
            module_dir,
            "dummy_ddp_app",
            """\
            import os
            import signal
            import sys
            import time
            from pathlib import Path

            signal_path = Path(os.environ["PYINE_TEST_SIGNAL_PATH"])
            ready_path = Path(os.environ["PYINE_TEST_READY_PATH"])

            def handle_sigint(signum, frame):
                signal_path.write_text(str(signum))
                sys.exit(130)

            signal.signal(signal.SIGINT, handle_sigint)
            ready_path.write_text("ready")
            print("dummy_ddp_app ready", flush=True)

            while True:
                time.sleep(0.1)
            """,
        )

        monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}")
        monkeypatch.setenv("PYTHONPATH", f"{module_dir}{os.pathsep}{os.environ.get('PYTHONPATH', '')}")
        monkeypatch.setenv("APP_MODULE", "dummy_ddp_app")
        monkeypatch.setenv("TEE_LOG", "1")
        monkeypatch.setenv("LOG_DIR", str(tmp_path / "logs"))
        monkeypatch.setenv("RUN_NAME", "pytest_signal_forwarding")
        monkeypatch.setenv("PYTHON_BIN", sys.executable)
        monkeypatch.setenv("NPROC_PER_NODE", "1")
        monkeypatch.setenv("PYINE_TEST_SIGNAL_PATH", str(signal_record_path))
        monkeypatch.setenv("PYINE_TEST_READY_PATH", str(ready_marker_path))

        stdout_data = ""
        proc = subprocess.Popen(
            [str(launcher_path)],
            cwd=repo_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        try:
            deadline = time.time() + 10
            while not ready_marker_path.exists():
                if time.time() > deadline:
                    raise AssertionError("dummy_ddp_app did not mark readiness before timeout")
                if proc.poll() is not None:
                    stdout_tail = proc.stdout.read() if proc.stdout else ""
                    raise AssertionError(f"launcher exited early: rc={proc.returncode}, output={stdout_tail}")
                time.sleep(0.1)

            proc.send_signal(signal.SIGINT)
            stdout_data, _ = proc.communicate(timeout=15)
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.communicate()

        assert proc.returncode == 130, stdout_data
        assert signal_record_path.exists(), "torchrun stub should record the forwarded SIGINT"
        assert signal_record_path.read_text().strip() == str(signal.SIGINT)

        log_path = tmp_path / "logs" / "launch_pytest_signal_forwarding.log"
        assert log_path.exists(), "launcher should append logs when tee is enabled"
        log_contents = log_path.read_text()
        assert "dummy_ddp_app ready" in log_contents
        assert "torchrun launch" in log_contents.lower(), log_contents

    def test_dry_run_outputs_summary(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Dry-run mode should print the launch summary without touching the filesystem."""
        repo_root = Path(__file__).resolve().parents[2]
        launcher_path = repo_root / "scripts" / "run_ddp.sh"

        bin_dir = self._install_torchrun_stub(tmp_path)
        module_dir = tmp_path / "fake_app"
        self._write_module(
            module_dir,
            "dummy_ddp_app",
            "def main():\n    pass\n",
        )

        monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}")
        monkeypatch.setenv("PYTHONPATH", f"{module_dir}{os.pathsep}{os.environ.get('PYTHONPATH', '')}")
        monkeypatch.setenv("APP_MODULE", "dummy_ddp_app")
        monkeypatch.setenv("TEE_LOG", "1")
        monkeypatch.setenv("LOG_DIR", str(tmp_path / "logs"))
        monkeypatch.setenv("RUN_NAME", "pytest_dry_run")
        monkeypatch.setenv("PYTHON_BIN", sys.executable)
        monkeypatch.setenv("NPROC_PER_NODE", "1")

        result = subprocess.run(
            [str(launcher_path), "--dry-run"],
            cwd=repo_root,
            check=False,
            text=True,
            capture_output=True,
        )

        assert result.returncode == 0, result.stdout + result.stderr
        assert "=== Launch Summary ===" in result.stdout
        assert "[dry-run] not executing." in result.stdout
        assert not (tmp_path / "logs").exists(), "dry-run should not create log directories"

    def test_manual_multinode_requires_master_addr(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Launching multi-node jobs outside SLURM should enforce a non-local master address."""
        repo_root = Path(__file__).resolve().parents[2]
        launcher_path = repo_root / "scripts" / "run_ddp.sh"

        bin_dir = self._install_torchrun_stub(tmp_path)
        monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}")
        monkeypatch.setenv("NPROC_PER_NODE", "1")
        monkeypatch.setenv("NNODES", "2")
        monkeypatch.delenv("SLURM_JOB_ID", raising=False)

        result = subprocess.run(
            [str(launcher_path), "--dry-run"],
            cwd=repo_root,
            check=False,
            text=True,
            capture_output=True,
        )

        assert result.returncode != 0
        assert "multi-node launch requires" in result.stdout

    def test_tee_zero_skips_log_file(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """When TEE_LOG=0 the launcher should forego log files and leave stdout intact."""
        repo_root = Path(__file__).resolve().parents[2]
        launcher_path = repo_root / "scripts" / "run_ddp.sh"

        bin_dir = self._install_torchrun_stub(tmp_path)
        module_dir = tmp_path / "fake_app"
        module_output_path = tmp_path / "module_output.txt"
        self._write_module(
            module_dir,
            "dummy_exit_app",
            """\
            import os
            from pathlib import Path

            Path(os.environ["PYINE_TEST_MODULE_OUTPUT"]).write_text("ran")
            print("dummy_exit_app ran", flush=True)
            """,
        )

        monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}")
        monkeypatch.setenv("PYTHONPATH", f"{module_dir}{os.pathsep}{os.environ.get('PYTHONPATH', '')}")
        monkeypatch.setenv("APP_MODULE", "dummy_exit_app")
        monkeypatch.setenv("TEE_LOG", "0")
        monkeypatch.setenv("LOG_DIR", str(tmp_path / "logs"))
        monkeypatch.setenv("RUN_NAME", "pytest_no_tee")
        monkeypatch.setenv("PYTHON_BIN", sys.executable)
        monkeypatch.setenv("NPROC_PER_NODE", "1")
        monkeypatch.setenv("PYINE_TEST_MODULE_OUTPUT", str(module_output_path))

        result = subprocess.run(
            [str(launcher_path)],
            cwd=repo_root,
            check=False,
            text=True,
            capture_output=True,
        )

        assert result.returncode == 0, result.stdout + result.stderr
        assert module_output_path.read_text() == "ran"
        assert "dummy_exit_app ran" in result.stdout
        log_path = tmp_path / "logs" / "launch_pytest_no_tee.log"
        assert not log_path.exists(), "TEE_LOG=0 should disable launcher log creation"
