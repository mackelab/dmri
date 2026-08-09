"""Execute the documentation notebooks and render them to Markdown.

Build tooling for the docs site. Run ``python docs/render_examples.py`` before
``zensical build``; the Markdown it writes is gitignored and regenerated every
time.

By default every notebook is re-executed, so a figure in the docs always
reflects the code that is in the tree and a broken example fails the build.
Notebooks listed in :data:`NOT_EXECUTED` are rendered from their committed
outputs instead -- use that for examples too slow or too network-dependent to
run in CI.
"""

import argparse
import re
import shutil
from pathlib import Path

import nbformat
from nbconvert import MarkdownExporter
from nbconvert.writers import FilesWriter

#: Notebooks rendered from committed outputs rather than re-executed.
#: `05_example_data` downloads 174 MB from a university host and runs a full
#: prediction; that belongs in a human's hands, not in every docs build.
NOT_EXECUTED = frozenset({"05_example_data"})

#: How long a single notebook may take before the build gives up.
EXECUTE_TIMEOUT_SECONDS = 900


def _exporter_config():
    """Exporter settings, as a traitlets Config.

    Must be a `Config` rather than a plain dict: a dict reaches the exporter's
    own traits but is silently ignored by nested preprocessors, which is how
    the SVG setting below can look applied while doing nothing.
    """
    from traitlets.config import Config

    config = Config()
    config.MarkdownExporter.exclude_input_prompt = True
    config.MarkdownExporter.exclude_output_prompt = True
    # nbconvert extracts PNG/JPEG/PDF by default and inlines anything else as a
    # mislabelled `data:image/svg;base64,<raw markup>` URI that no browser
    # renders. Notebook 02 draws hand-built SVG, so without this its figures
    # vanish from the site with no warning.
    config.ExtractOutputPreprocessor.extract_output_types = {
        "image/png",
        "image/jpeg",
        "image/svg+xml",
        "application/pdf",
    }
    return config


def execute(notebook, notebook_path: Path):
    """Run every cell, leaving the outputs on the in-memory notebook."""
    from nbclient import NotebookClient

    client = NotebookClient(
        notebook,
        timeout=EXECUTE_TIMEOUT_SECONDS,
        kernel_name="python3",
        resources={"metadata": {"path": str(notebook_path.parent)}},
    )
    client.execute()
    return notebook


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="dmri._docs_render", description=__doc__)
    parser.add_argument(
        "--no-execute",
        action="store_true",
        help="Render committed outputs without running any notebook.",
    )
    args = parser.parse_args(argv)

    examples_dir = Path(__file__).parent / "examples"

    for notebook_path in sorted(examples_dir.glob("[0-9][0-9]_*.ipynb")):
        output_files_dir = f"{notebook_path.stem}_files"
        shutil.rmtree(examples_dir / output_files_dir, ignore_errors=True)

        notebook = nbformat.read(notebook_path, as_version=4)
        nbformat.validate(notebook)

        skipped = args.no_execute or notebook_path.stem in NOT_EXECUTED
        print(f"{'rendering' if skipped else 'executing'} {notebook_path.name}")
        if not skipped:
            notebook = execute(notebook, notebook_path)
        for cell in notebook.cells:
            if cell.cell_type == "code":
                cell.outputs = [
                    output
                    for output in cell.outputs
                    if not (output.output_type == "stream" and output.name == "stderr")
                ]
        exporter = MarkdownExporter(config=_exporter_config())
        body, resources = exporter.from_notebook_node(
            notebook,
            resources={"output_files_dir": output_files_dir},
        )
        body = add_figure_alt_text(body)
        title, separator, content = body.partition("\n")
        body = (
            "<!-- Generated from the adjacent notebook. Do not edit directly. -->\n\n"
            f"{title}{separator}\n"
            f"[Download notebook]({notebook_path.name}){{ .md-button }}\n\n"
            f"{content.lstrip()}"
        )
        FilesWriter(build_directory=str(examples_dir)).write(
            body,
            resources,
            notebook_name=notebook_path.stem,
        )


def add_figure_alt_text(markdown: str) -> str:
    """Describe exported figures using their nearest section heading."""
    heading = "Notebook output"
    lines = []
    for line in markdown.splitlines():
        if line.startswith("## "):
            heading = re.sub(r"[`*_]", "", line.removeprefix("## "))
        line = re.sub(r"!\[(?:png|svg)\]", f"![{heading}]", line)
        lines.append(line)
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
