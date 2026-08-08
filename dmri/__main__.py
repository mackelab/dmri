"""Entry point for `python -m dmri`.

Without this, `python -m dmri` fails with "cannot be directly executed", and the
individual script modules exit 0 having done nothing -- a silent no-op that
looks like success. Mirrors the `dmri` console script in pyproject.toml.
"""

from dmri.cli import main

if __name__ == "__main__":
    main()
