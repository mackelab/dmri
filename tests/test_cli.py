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
