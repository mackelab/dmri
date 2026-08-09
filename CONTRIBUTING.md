# Contributing

Contributions are welcome through GitHub issues and pull requests.

## Development setup

DMRI requires Python 3.11 or newer and uses [uv](https://docs.astral.sh/uv/)
for development environments.

```bash
uv venv -p 3.11
source .venv/bin/activate
uv pip install -e '.[dev,docs]'
```

## Checks

Run the test suite:

```bash
uv run pytest
```

Check linting and formatting:

```bash
uv run ruff check .
uv run ruff format --check .
```

Apply automatic fixes and formatting with `uv run ruff check --fix .` and
`uv run ruff format .`.

## Documentation

Execute and render the example notebooks before previewing the documentation:

```bash
uv run --extra docs python docs/render_examples.py
uv run --extra docs zensical serve
```

Every notebook is re-executed so its figures match the code in the tree. Pass
`--no-execute` to render the committed outputs instead, which is much faster
while you are editing prose.

Keep changes focused, add tests for changed behavior, and update documentation
when user-facing behavior changes.
