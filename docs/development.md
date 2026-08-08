# Development

Install the development and documentation dependencies from a repository clone:

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

Run formatting, lint, and repository hooks:

```bash
uv run ruff format --check dmri tests
uv run ruff check dmri tests
uv run pre-commit run --all-files
```

Build the documentation:

```bash
uv run --extra docs zensical build
```

The build output is under `site/`. A documentation change should not introduce
missing pages or broken links.

## Examples

Example notebooks are stored in `docs/examples/`. Execute a notebook without
overwriting it with:

```bash
uv run --extra docs jupyter nbconvert \
  --execute --to notebook --stdout docs/examples/04_train_standalone.ipynb \
  >/dev/null
```

Refresh saved outputs only when the outputs are part of the documentation:

```bash
uv run --extra docs jupyter nbconvert \
  --execute --to notebook --inplace docs/examples/04_train_standalone.ipynb
```

## Documentation conventions

- Prefer runnable commands and concrete file layouts.
- Keep workflow explanation in guides and exact option lists in CLI pages.
- Document only Python functions intended for reuse.
- State hardware and data assumptions.
- Do not claim exact reproducibility from a saved config alone.
- Update examples when an API or configuration schema changes.
