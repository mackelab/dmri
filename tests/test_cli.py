import os
import subprocess
import sys

import pytest

from dmri import cli


def test_cli_prints_help_without_command(capsys, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["dmri"])
    cli.main()
    assert "dmri <command>" in capsys.readouterr().out


def test_cli_rejects_unknown_command(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["dmri", "unknown"])
    with pytest.raises(SystemExit, match="2"):
        cli.main()


def test_cli_sets_native_log_policy_before_jax_imports():
    env = os.environ.copy()
    env.pop("DMRI_SHOW_NATIVE_LOGS", None)
    env.pop("TF_CPP_MIN_LOG_LEVEL", None)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import os, sys; import dmri.cli; "
                "print(os.environ['TF_CPP_MIN_LOG_LEVEL']); "
                "print('jax' in sys.modules)"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.stdout.splitlines() == ["3", "False"]


def test_native_logs_have_an_explicit_escape_hatch():
    env = os.environ.copy()
    env["DMRI_SHOW_NATIVE_LOGS"] = "1"
    env["TF_CPP_MIN_LOG_LEVEL"] = "1"
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import os; import dmri.cli; print(os.environ['TF_CPP_MIN_LOG_LEVEL'])",
        ],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.stdout.strip() == "1"
