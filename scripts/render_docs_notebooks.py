"""Render documentation notebooks to Markdown for Zensical."""

import re
import shutil
from pathlib import Path

import nbformat
from nbconvert import MarkdownExporter
from nbconvert.writers import FilesWriter


def main() -> None:
    examples_dir = Path(__file__).parents[1] / "docs" / "examples"

    for notebook_path in sorted(examples_dir.glob("[0-9][0-9]_*.ipynb")):
        output_files_dir = f"{notebook_path.stem}_files"
        shutil.rmtree(examples_dir / output_files_dir, ignore_errors=True)

        notebook = nbformat.read(notebook_path, as_version=4)
        nbformat.validate(notebook)
        for cell in notebook.cells:
            if cell.cell_type == "code":
                cell.outputs = [
                    output
                    for output in cell.outputs
                    if not (output.output_type == "stream" and output.name == "stderr")
                ]
        exporter = MarkdownExporter(
            config={
                "MarkdownExporter": {
                    "exclude_input_prompt": True,
                    "exclude_output_prompt": True,
                }
            }
        )
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
