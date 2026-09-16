#!/usr/bin/env python3
"""
Static check on the tutorial notebooks. Run this before a class.

    python check_notebooks.py

It does not execute anything, so it takes under a second. It checks that
every notebook parses, that the exercise notebooks still carry their blanks,
that the answer notebooks carry none, and that the import surface is the one
the requirements file installs. To check that the notebooks actually *run*,
execute them; the three exercises take about ten minutes each on a CPU.

The site pages and the slide deck are checked by predeploy.py.
"""
import ast
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).parent
EXERCISES = {"3_activity_conformers", "6_activity_ir_spectra", "7_activity_reactivity"}
ALLOWED_TOP_LEVEL = {
    "subprocess", "sys", "warnings", "numpy", "torch", "aimnet", "ase",
    "rdkit", "matplotlib", "sella", "IPython", "itertools",
    "math", "collections", "time", "os",
}
# fallback for cells that ast cannot parse (IPython magics, shell escapes)
IMPORT_RE = re.compile(r"^\s*(?:import\s+([^#\n]+)|from\s+([\w.]+)\s+import\b)", re.M)

ok = True


def fail(msg):
    global ok
    ok = False
    print(f"  FAIL  {msg}")


def imported_modules(src):
    """Top-level names of every module a code cell imports, at any indentation."""
    mods = set()
    try:
        tree = ast.parse(src)
    except SyntaxError:
        for plain, from_ in IMPORT_RE.findall(src):
            names = [n.split(" as ")[0] for n in plain.split(",")] if plain else [from_]
            mods.update(n.strip().split(".")[0] for n in names if n.strip())
        return mods
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            mods.add(node.module.split(".")[0])
    return mods


def check(folder, want_todo):
    print(f"\n{folder}/")
    paths = sorted((ROOT / folder).glob("*.ipynb"))
    if not paths:
        fail(f"no notebooks found in {folder}/")
        return
    for path in paths:
        try:
            nb = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            fail(f"{path.name}: not valid JSON ({exc})")
            continue

        cells = nb["cells"]
        code = [c for c in cells if c["cell_type"] == "code"]
        md = [c for c in cells if c["cell_type"] == "markdown"]
        src = "\n".join("".join(c["source"]) for c in cells)
        n_todo = len(re.findall(r"# TODO", src))
        stem = path.stem.replace("_ANSWERS", "")

        if want_todo and stem in EXERCISES and n_todo == 0:
            fail(f"{path.name}: exercise has no blanks left for students")
        if not want_todo and n_todo:
            fail(f"{path.name}: answer notebook still has {n_todo} TODO marker(s)")
        if any(c.get("outputs") for c in code):
            fail(f"{path.name}: stored outputs, clear them before committing")
        if not md:
            fail(f"{path.name}: no prose")

        # Imports are only checked in code cells. Markdown may quote an
        # import from a package the tutorial does not install, as notebook 8
        # does for openmmml.
        found = set()
        for c in code:
            found |= imported_modules("".join(c["source"]))
        for mod in sorted(found - ALLOWED_TOP_LEVEL):
            fail(f"{path.name}: imports '{mod}', which requirements.txt does not install")

        print(f"  ok    {path.name:<42s} {len(code):2d} code  {len(md):2d} md  {n_todo} TODO")


check("notebooks", want_todo=True)
check("solutions", want_todo=False)

print("\nall checks passed" if ok else "\nsome checks failed")
sys.exit(0 if ok else 1)
