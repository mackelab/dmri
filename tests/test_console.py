import pytest

from dmri import console


@pytest.fixture(autouse=True)
def reset_console():
    console.configure(enabled=False)
    yield
    console.configure(enabled=False)


def test_console_is_quiet_until_enabled(capsys):
    console.say("hidden")
    assert capsys.readouterr().out == ""

    console.configure(enabled=True, verbose=True)
    console.say("visible %s", "line")
    assert console.verbose()
    assert capsys.readouterr().out == "visible line\n"


def test_summary_is_plain_text_when_redirected(capsys):
    console.configure(enabled=True)
    console.summary("Prediction", [("Input", "/data/a"), ("Output", "/data/b")])
    captured = capsys.readouterr()
    assert "Prediction" in captured.out
    assert "Input" in captured.out and "/data/a" in captured.out
    assert "\x1b[" not in captured.out


def test_warning_is_written_to_stderr(capsys):
    console.configure(enabled=True)
    console.warning("invalid samples will be NaN")
    assert capsys.readouterr().err == "Warning: invalid samples will be NaN\n"


def test_non_terminal_progress_has_stable_start_and_finish(capsys):
    console.configure(enabled=True)
    progress = console.voxel_progress("Sampling", 10)
    progress.update(10)
    progress.close()

    assert capsys.readouterr().err.splitlines() == [
        "Sampling: 0/10 vox",
        "Sampling: 10/10 vox complete",
    ]


def test_non_terminal_progress_reports_an_interrupted_run(capsys):
    console.configure(enabled=True)
    progress = console.voxel_progress("Sampling", 10)
    progress.update(4)
    progress.close()

    assert capsys.readouterr().err.splitlines()[-1].endswith("4/10 vox stopped")
