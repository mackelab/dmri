"""User-facing console output, separate from the diagnostic log.

The evaluation log is verbose by design and is what you want in a file. A CLI run
wants a handful of lines and some progress bars. Keeping the two apart means the
log can be silenced without also silencing the things a person is waiting to see.
"""

import sys

_enabled = False


def set_enabled(enabled: bool) -> None:
    """Turn console output on, which the CLI does and library callers do not."""
    global _enabled
    _enabled = bool(enabled)


def enabled() -> bool:
    return _enabled


def say(message: str = "", *args) -> None:
    """Write one line for the person running the command."""
    if not _enabled:
        return
    print(message % args if args else message, file=sys.stdout, flush=True)
