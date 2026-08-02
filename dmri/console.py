"""Consistent user-facing terminal output for DMRI commands."""

import sys
from contextlib import contextmanager
from pathlib import Path

from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from rich.text import Text

LOGO = r""" /$$$$$$$  /$$      /$$ /$$$$$$$  /$$$$$$
| $$__  $$| $$$    /$$$| $$__  $$|_  $$_/
| $$  \ $$| $$$$  /$$$$| $$  \ $$  | $$
| $$  | $$| $$ $$/$$ $$| $$$$$$$/  | $$
| $$  | $$| $$  $$$| $$| $$__  $$  | $$
| $$  | $$| $$\  $ | $$| $$  \ $$  | $$
| $$$$$$$/| $$ \/  | $$| $$  | $$ /$$$$$$
|_______/ |__/     |__/|__/  |__/|______/"""

_enabled = False
_verbose = False
_output = Console(file=sys.stdout, highlight=False)
_activity = Console(file=sys.stderr, highlight=False)


def configure(*, enabled: bool, verbose: bool = False) -> None:
    """Configure output for a command invocation."""
    global _activity, _enabled, _output, _verbose
    _enabled = bool(enabled)
    _verbose = bool(verbose)
    # Test capture and embedding applications can replace the standard streams.
    _output = Console(file=sys.stdout, highlight=False)
    _activity = Console(file=sys.stderr, highlight=False)


def set_enabled(enabled: bool) -> None:
    """Compatibility shorthand for library and CLI callers."""
    configure(enabled=enabled, verbose=_verbose)


def enabled() -> bool:
    """Whether command output is currently enabled."""
    return _enabled


def verbose() -> bool:
    """Whether detailed (verbose) output is requested."""
    return _verbose


def print_logo() -> None:
    """Print the application logo before command-specific output."""
    print(LOGO, file=sys.stdout, flush=True)


def say(message: str = "", *args) -> None:
    """Write one stable line to stdout."""
    if _enabled:
        _output.print(Text(message % args if args else message))


def link(label: str, target) -> None:
    """Write a local path as a clickable link on one line.

    Terminals only linkify a URI that is not broken across lines, so this
    bypasses the usual wrapping. Where OSC 8 hyperlinks are supported the path
    becomes directly clickable; elsewhere the plain file:// URI is still
    recognised by most terminals.
    """
    if not _enabled:
        return
    uri = Path(target).resolve().as_uri()
    text = Text(f"{label}{uri}")
    text.stylize(f"link {uri}", len(label))
    _output.print(text, no_wrap=True, overflow="ignore", crop=False)


def warning(message: str) -> None:
    """Write an actionable warning without disturbing live terminal output."""
    if _enabled:
        _activity.print(Text(f"Warning: {message}", style="yellow"))


def summary(title: str, rows) -> None:
    """Render a compact command summary."""
    if not _enabled:
        return
    table = Table.grid(padding=(0, 3))
    table.add_column(style="bold")
    table.add_column(overflow="fold")
    for label, value in rows:
        table.add_row(Text(label), Text(str(value)))
    _output.print()
    _output.print(Text(title, style="bold"))
    _output.print(table)
    _output.print()


@contextmanager
def status(message: str):
    """Show a spinner on a terminal and a stable start line otherwise."""
    if not _enabled:
        yield
        return
    if not _activity.is_terminal:
        _activity.print(Text(message))
        yield
        return
    with _activity.status(Text(message), spinner="dots"):
        yield


class _PlainProgress:
    def __init__(self, description, total):
        self.description = description
        self.total = total
        self.completed = 0
        self.closed = False
        _activity.print(Text(f"{description}: 0/{total:,} vox"))

    def update(self, advance):
        self.completed += advance

    def reset(self):
        self.completed = 0

    def close(self):
        if not self.closed:
            state = "complete" if self.completed >= self.total else "stopped"
            _activity.print(
                Text(
                    f"{self.description}: {self.completed:,}/{self.total:,} vox {state}"
                )
            )
            self.closed = True


class _RichProgress:
    def __init__(self, description, total):
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("{task.description}"),
            BarColumn(bar_width=24),
            TaskProgressColumn(),
            TextColumn("{task.completed:,.0f}/{task.total:,.0f} vox"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=_activity,
        )
        self.progress.start()
        self.task_id = self.progress.add_task(description, total=total)
        self.closed = False

    def update(self, advance):
        self.progress.update(self.task_id, advance=advance)

    def reset(self):
        self.progress.reset(self.task_id)

    def close(self):
        if not self.closed:
            self.progress.stop()
            self.closed = True


def voxel_progress(description: str, total: int):
    """Create voxel progress appropriate for the current output stream."""
    if not _enabled or not description:
        return None
    if _activity.is_terminal:
        return _RichProgress(description, total)
    return _PlainProgress(description, total)
