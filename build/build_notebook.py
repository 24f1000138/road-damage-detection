"""Builds the Kaggle notebooks from their `# %%`-delimited sources.

    python build/build_notebook.py

svrdd_detection.ipynb  <- notebook_src.py   (stage 1: detection)
svrdd_priority.ipynb   <- priority_src.py   (stage 2: maintenance priority)

In stage 1, code cells from section 6 onwards get the `%%stage` cell magic (defined in its
configuration cell), which skips them once a training cell has paused for the session time budget.
In stage 2, a cell headed `# %% include=<tag>` is replaced by the stage-1 cells carrying that tag,
so both notebooks share one copy of the evaluation, fusion and dataset-parsing code.
"""
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "notebook_src.py"
OUT = HERE.parent / "svrdd_detection.ipynb"
SRC2 = HERE / "priority_src.py"
OUT2 = HERE.parent / "svrdd_priority.ipynb"


def parse_cells(text=None, stage_from=r"## 6\."):
    """Returns a list of dicts: {type: 'markdown'|'code', source: str, tags: set, stage: bool, include: str|None}."""
    text = SRC.read_text(encoding="utf-8") if text is None else text
    cells, cur = [], None
    staged = False
    for line in text.splitlines():
        if line.startswith("# %%"):
            if cur:
                cells.append(cur)
            head = line[4:].strip()
            kind = "markdown" if head.startswith("[markdown]") else "code"
            tags = set(re.findall(r"tags=(\w+)", head))
            inc = re.findall(r"include=(\w+)", head)
            cur = dict(type=kind, lines=[], tags=tags, include=inc[0] if inc else None)
            continue
        if cur is None:
            continue
        cur["lines"].append(line)
    if cur:
        cells.append(cur)
    for c in cells:
        if c["type"] == "markdown":
            c["source"] = "\n".join(l[2:] if l.startswith("# ") else l.lstrip("#") for l in c["lines"]).strip()
            if stage_from and re.match(stage_from, c["source"]):
                staged = True
        else:
            c["source"] = "\n".join(c["lines"]).strip()
        c["stage"] = staged and c["type"] == "code"
        del c["lines"]
    return cells


SRC3 = HERE / "demo_src.py"
OUT3 = HERE.parent / "svrdd_demo.ipynb"


def parse_priority_cells(src=None):
    """Cells of a dependent notebook (priority or demo) with `include=` directives resolved against the
    stage-1 and stage-2 sources."""
    shared = parse_cells() + parse_cells(SRC2.read_text(encoding="utf-8"), stage_from=None)
    cells = parse_cells((src or SRC2).read_text(encoding="utf-8"), stage_from=None)
    for c in cells:
        if c["include"]:
            parts = [s["source"] for s in shared if c["include"] in s["tags"]]
            assert parts, f"no stage-1 cell is tagged {c['include']!r}"
            c["source"] = f"# Shared with the detection notebook (tag: {c['include']}).\n" + "\n\n".join(parts)
    return cells


def write_notebook(cells, out, accelerator):
    nb_cells = []
    for c in cells:
        src = c["source"]
        if c["type"] == "code":
            compile(src, "<cell>", "exec")                       # fail the build on a syntax error
            if c["stage"]:
                src = "%%stage\n" + src
            nb_cells.append(dict(cell_type="code", metadata={}, execution_count=None, outputs=[],
                                 source=src.splitlines(keepends=True)))
        else:
            nb_cells.append(dict(cell_type="markdown", metadata={}, source=src.splitlines(keepends=True)))
    kaggle = dict(isInternetEnabled=True, isGpuEnabled=bool(accelerator))
    if accelerator:
        kaggle["accelerator"] = accelerator
    nb = dict(cells=nb_cells, nbformat=4, nbformat_minor=5,
              metadata=dict(kernelspec=dict(display_name="Python 3", language="python", name="python3"),
                            language_info=dict(name="python"), kaggle=kaggle))
    out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    n_code = sum(c["cell_type"] == "code" for c in nb_cells)
    print(f"wrote {out} ({len(nb_cells)} cells, {n_code} code)")


def build():
    write_notebook(parse_cells(), OUT, "nvidiaTeslaT4")
    if SRC2.exists():
        write_notebook(parse_priority_cells(), OUT2, None)
    if SRC3.exists():
        write_notebook(parse_priority_cells(SRC3), OUT3, "nvidiaTeslaT4")


if __name__ == "__main__":
    build()
