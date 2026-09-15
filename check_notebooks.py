#!/usr/bin/env python3
"""
Static check on the tutorial notebooks. Run this before a class.

    python check_notebooks.py

It does not execute anything, so it takes under a second. It checks that
every notebook parses, that the exercise notebooks still carry their blanks,
that the answer notebooks carry none, and that the import surface is the one
the requirements file installs. To check that the notebooks actually *run*,
execute them; the three exercises take about ten minutes each on a CPU.
"""
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

ok = True


def fail(msg):
    global ok
    ok = False
    print(f"  FAIL  {msg}")


def check(folder, want_todo):
    print(f"\n{folder}/")
    paths = sorted((ROOT / folder).glob("*.ipynb"))
    if not paths:
        fail(f"no notebooks found in {folder}/")
        return
    for path in paths:
        try:
            nb = json.loads(path.read_text())
        except json.JSONDecodeError as exc:
            fail(f"{path.name}: not valid JSON ({exc})")
            continue

        cells = nb["cells"]
        code = [c for c in cells if c["cell_type"] == "code"]
        md = [c for c in cells if c["cell_type"] == "markdown"]
        src = "\n".join("".join(c["source"]) for c in cells)
        code_src = "\n".join("".join(c["source"]) for c in code)
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
        for mod in re.findall(r"^(?:import|from)\s+([A-Za-z_][\w]*)", code_src, re.M):
            if mod not in ALLOWED_TOP_LEVEL:
                fail(f"{path.name}: imports '{mod}', which requirements.txt does not install")

        print(f"  ok    {path.name:<42s} {len(code):2d} code  {len(md):2d} md  {n_todo} TODO")


check("notebooks", want_todo=True)
check("solutions", want_todo=False)

for page in ("index.html", "install.html", "notebooks.html",
             "troubleshooting.html", "teaching.html"):
    if not (ROOT / page).exists():
        fail(f"{page} is missing")
print("\nsite/")
html_pages = [p for p in ("index.html", "install.html", "notebooks.html",
                          "troubleshooting.html", "teaching.html")
              if (ROOT / p).exists()]
targets = set(html_pages) | {"slides/"}
for page in html_pages:
    html = (ROOT / page).read_text()
    for href in re.findall(r'href="([^"#?]+)', html):
        if href.startswith(("http://", "https://", "mailto:")):
            continue
        if href not in targets and not (ROOT / href).exists():
            fail(f"{page}: dead link to {href}")
    print(f"  ok    {page:<42s} {len(html) // 1024} KB")

slides = ROOT / "slides" / "index.html"
if not slides.exists():
    fail("slides/index.html is missing")
else:
    html = slides.read_text()
    n = html.count('class="slide')
    print(f"\nslides/\n  ok    index.html{'':<32s} {n} slides  {len(html) // 1024} KB")
    # http://www.w3.org/2000/svg is an XML namespace, not a network load.
    insecure = [u for u in re.findall(r'(?:src|href)="(http://[^"]+)"', html)
                if not u.startswith("http://www.w3.org/")]
    if insecure:
        fail(f"slides/index.html loads {insecure[0]} over plain http")

print("\nall checks passed" if ok else "\nsome checks failed")
sys.exit(0 if ok else 1)
