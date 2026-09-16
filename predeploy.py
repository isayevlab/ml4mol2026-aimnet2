#!/usr/bin/env python3
"""
Pre-deployment check. Run before pushing to GitHub.

    python predeploy.py

Verifies that every link resolves, that no notebook carries stored outputs, that
no absolute local path leaked into a file, and that prose follows the house
style. It does not execute anything; see run.py for that.
"""
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).parent
REPO = "isayevlab/ml4mol2026-aimnet2"
PAGES = ["index.html", "install.html", "notebooks.html",
         "troubleshooting.html", "teaching.html"]
# section headers deliberately use an em dash: "Definition — x", "Step 1 — x"
HEADER = re.compile(r"(Definition|Definitions|Reminder|Exercise|Step)s?\s*\d*\s*\u2014")

bad = []


def check(condition, message):
    if not condition:
        bad.append(message)


# 1. links inside the site resolve to a file or to a known directory
targets = set(PAGES) | {"slides/"}
for name in PAGES:
    page = ROOT / name
    check(page.exists(), f"{name} is missing")
    if not page.exists():
        continue
    html = page.read_text(encoding="utf-8")
    for href in re.findall(r'href="([^"#?]+)', html):
        if href.startswith(("http://", "https://", "mailto:")):
            continue
        check(href in targets or (ROOT / href).exists(),
              f"{name}: dead link to {href}")

# 2. every Colab and blob link points at a notebook that exists
for f in list(ROOT.glob("*.html")) + [ROOT / "README.md"]:
    text = f.read_text(encoding="utf-8")
    for url in re.findall(rf"(?:colab\.research\.google\.com/github/{REPO}|"
                          rf"github\.com/{REPO})/blob/main/(\S+?\.ipynb)", text):
        check((ROOT / url).exists(), f"{f.name}: link to missing {url}")

# 3. notebooks are valid JSON, carry no outputs, and keep their blanks
for nb_path in sorted(ROOT.glob("*/*.ipynb")):
    try:
        nb = json.loads(nb_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        bad.append(f"{nb_path}: invalid JSON ({exc})")
        continue
    rel = nb_path.relative_to(ROOT)
    check(not any(c.get("outputs") for c in nb["cells"]),
          f"{rel}: stored outputs, clear them before committing")
    check(not any(c.get("execution_count") for c in nb["cells"]),
          f"{rel}: execution counts left in")
    n_todo = json.dumps(nb).count("# TODO")
    if "activity" in nb_path.stem and "ANSWERS" not in nb_path.stem:
        check(n_todo > 0, f"{rel}: exercise has no blanks left for students")
    if "ANSWERS" in nb_path.stem:
        check(n_todo == 0, f"{rel}: answer notebook still has {n_todo} TODO marker(s)")

# 4. nothing leaks an absolute local path
# A Windows profile path counts only when it names a real user; the install
# guide legitimately shows the placeholder C:\Users\<you>\...
LEAKS = (re.escape("/home/" + "claude"), re.escape("/mnt/" + "user-data"),
         r"C:[\\/]Users[\\/](?![<&$%{])")
for f in [p for pat in ("*.py", "*.ipynb", "*.html", "*.md", "*.txt")
          for p in ROOT.rglob(pat)]:
    rel = f.relative_to(ROOT)
    if f.name == "predeploy.py":
        continue                       # this file names the patterns it looks for
    if any(part.startswith(".") for part in rel.parts):
        continue                       # .venv/, .git/ and other untracked dot-dirs
    text = f.read_text(encoding="utf-8", errors="ignore")
    for leak in LEAKS:
        m = re.search(leak, text)
        check(m is None, f"{rel}: contains local path {m.group(0) if m else ''}")

# 5. house style in notebook prose
for nb_path in sorted((ROOT / "notebooks").glob("*.ipynb")):
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "markdown":
            continue
        for line in "".join(cell["source"]).split("\n"):
            if "\u2014" in line and not HEADER.search(line):
                bad.append(f"{nb_path.name} cell {i}: em dash in prose: "
                           f"{line.strip()[:70]}")

# 6. the deck loads nothing over plain http
slides = ROOT / "slides" / "index.html"
check(slides.exists(), "slides/index.html is missing")
if slides.exists():
    html = slides.read_text(encoding="utf-8")
    insecure = [u for u in re.findall(r'(?:src|href)="(http://[^"]+)"', html)
                if not u.startswith("http://www.w3.org/")]
    check(not insecure, f"slides/index.html loads {insecure[:1]} over plain http")

# 7. GitHub Pages needs these at the repository root
for required in (".nojekyll", "index.html", "slides/index.html"):
    check((ROOT / required).exists(), f"{required} is required for GitHub Pages")

if bad:
    print("\n".join(f"  FAIL  {b}" for b in bad))
    print(f"\n{len(bad)} problem(s) found")
    sys.exit(1)
print("pre-deployment checks: all clear")
